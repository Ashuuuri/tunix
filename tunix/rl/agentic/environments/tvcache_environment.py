"""Read-only SkyRL-SQL environment using a per-rollout TVCache executor.

Requires the team repository on PYTHONPATH (skyrl_integration package).
All TVClient and SQLite work stays on one private event-loop thread.
"""
import asyncio
import concurrent.futures
import json
from pathlib import Path
import threading
import uuid

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult
from skyrl_integration.skyrl_sql import SqlToolCall, load_task, sandbox_for
from tvclient import TvcacheToolExecutor


class TVCacheEnv(BaseTaskEnv):
    """Accepts ToolAgent function calls named `sql` and terminal `finish`.

    task_path identifies a prepared task.json. Reference SQL is never included
    in the initial observation. reward_fn is optional; without it rewards are 0
    and no correctness score is implied. One instance per rollout, no fork bank.
    """

    def __init__(self, task=None, *, task_path=None, tvcache_host='127.0.0.1',
                 tvcache_port=7878, max_steps=10, reward_fn=None,
                 operation_timeout=30.0, query_timeout=5.0, max_rows=10000,
                 cache_namespace='', use_cache=True, grade_final_sql=False, **kwargs):
        config = dict(task or {})
        self.task_path = Path(task_path or config['task_path']).resolve()
        if max_steps < 1 or operation_timeout <= 0 or query_timeout <= 0 or max_rows < 1:
            raise ValueError('Step, timeout, and row limits must be positive')
        super().__init__(task=config, reward_fn=reward_fn, max_steps=max_steps, **kwargs)
        self.host, self.port = tvcache_host, tvcache_port
        self.operation_timeout = operation_timeout
        self.query_timeout, self.max_rows = query_timeout, max_rows
        self.cache_namespace = cache_namespace
        self.use_cache = use_cache
        self.grade_final_sql = grade_final_sql
        self._direct_sandbox = None
        self._grading = None
        self._lock = threading.RLock()
        self._loop = self._thread = self._executor = None
        self.history = []
        self._finished = True
        self._stats = {'calls': 0, 'real_executions': 0, 'cache_hits': 0}

    def _start_loop(self):
        ready = threading.Event()
        def worker():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            try:
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
        self._thread = threading.Thread(target=worker, name='tvcache-env-' + uuid.uuid4().hex[:8], daemon=True)
        self._thread.start()
        ready.wait()

    def _run(self, coroutine):
        async def bounded():
            return await asyncio.wait_for(coroutine, timeout=self.operation_timeout)
        return asyncio.run_coroutine_threadsafe(bounded(), self._loop).result()

    async def _open(self):
        if not self.use_cache:
            self._direct_sandbox = self._sandbox_class(task_name=self.task_id)
            await self._direct_sandbox.start()
            return
        self._executor = TvcacheToolExecutor(
            self._sandbox_class, SqlToolCall, task_id=self.task_id,
            host=self.host, port=self.port)
        await self._executor.__aenter__()

    async def _shutdown(self):
        sandbox, self._direct_sandbox = self._direct_sandbox, None
        if sandbox is not None:
            await sandbox.stop()
        executor, self._executor = self._executor, None
        if executor is not None:
            await executor.close()

    def _release(self):
        if self._thread is None:
            return
        try:
            self._run(self._shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop = self._thread = None

    def reset(self):
        with self._lock:
            self._release()
            self._finished = True
            self.history = []
            self._stats = {'calls': 0, 'real_executions': 0, 'cache_hits': 0}
            self._grading = None
            self._record, database, identity = load_task(self.task_path)
            # Limits affect cached success/failure behavior and belong in identity.
            self.task_id = f'{identity}:rows={self.max_rows}:query_timeout={self.query_timeout}:{self.cache_namespace}'
            self._sandbox_class = sandbox_for(database, timeout=self.query_timeout, max_rows=self.max_rows)
            self._start_loop()
            try:
                self._run(self._open())
            except BaseException:
                self._release()
                raise
            self._finished = False
            return super().reset()

    def _initial_observation(self):
        # ToolAgent consumes question; include schema there, not just as a spare key.
        return {'question': self._record['question'] + '\n\nDatabase schema:\n' + self._record['schema'],
                'db_id': self._record['db_id']}

    @staticmethod
    def _normalize(action):
        action = getattr(action, 'action', action)
        if isinstance(action, str):
            return action, []
        if isinstance(action, dict):
            action = [action]
        if not isinstance(action, list):
            raise ValueError('Expected a final-answer string or function-call list')
        normalized = []
        for call in action:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                raise ValueError('Expected ToolAgent function-call dictionaries')
            function = call['function']
            name, arguments = function.get('name'), function.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object')
            if name == 'sql':
                SqlToolCall(arguments.get('sql'))  # validate entire batch before execution
            elif name == 'finish':
                if not isinstance(arguments.get('response', ''), str):
                    raise ValueError('finish.response must be a string')
            else:
                raise ValueError(f'Unsupported tool: {name!r}; expected sql or finish')
            normalized.append((call.get('id') or uuid.uuid4().hex, name, arguments))
        ids = [item[0] for item in normalized]
        if len(set(ids)) != len(ids):
            raise ValueError('Tool call IDs must be unique within a step')
        if any(name == 'finish' for _, name, _ in normalized):
            if len(normalized) != 1:
                raise ValueError('finish must be the only call in a step')
            return normalized[0][2].get('response', ''), []
        return None, normalized

    async def _execute(self, calls):
        outputs = {}
        for call_id, _, arguments in calls:
            call = SqlToolCall(arguments['sql'])
            history = self.history + [call]
            if self.use_cache:
                before = self._executor.total_executions
                value = await self._executor.lookup_or_execute(history)
                executed = self._executor.total_executions - before
            else:
                value = await self._direct_sandbox.execute(call)
                executed = 1
            self.history.append(call)
            self._stats['calls'] += 1
            self._stats['real_executions'] += executed
            self._stats['cache_hits'] += int(executed == 0)
            outputs[call_id] = value
        return outputs

    def _step_impl(self, action):
        answer, calls = self._normalize(action)
        if answer is not None:
            # Reward function is responsible for assessing the final response.
            if self.reward_fn is not None:
                reward = float(self.reward_fn(task=self._record, action=answer))
            elif self.grade_final_sql:
                from skyrl_integration.sql_reward import grade_sql
                self._grading = self._run(grade_sql(self._sandbox_class, self.task_id, self._record, answer))
                reward = self._grading['reward']
            else:
                reward = 0.0
            return EnvStepResult({}, reward, True, {'response': action, 'final_answer': answer,
                                 'tvcache': dict(self._stats), 'grading': self._grading})
        outputs = self._run(self._execute(calls))
        return EnvStepResult({'tool_outputs': outputs}, 0.0, False,
                             {'response': action, 'tvcache': dict(self._stats)})

    def step(self, action):
        with self._lock:
            if self._finished:
                raise RuntimeError('Call reset() before step(); episode has ended or is closed')
            try:
                result = super().step(action)
                if result[2]:
                    self._finished = True
                    self._release()
                return result
            except BaseException:
                self._finished = True
                self._release()
                raise

    def close(self):
        with self._lock:
            self._finished = True
            self._release()

    @classmethod
    def from_dict(cls, env_args):
        return cls(**dict(env_args))
