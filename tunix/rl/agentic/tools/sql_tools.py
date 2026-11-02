"""ToolAgent schemas: execution is deliberately owned by TVCacheEnv."""
from tunix.rl.agentic.tools.base_tool import BaseTool, ToolOutput


class SqlTool(BaseTool):
    """Execute one read-only SQLite SELECT/WITH query and inspect its result."""
    def get_json_schema(self):
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description,
                'parameters': {'type': 'object', 'properties': {'sql': {'type': 'string', 'description': 'One read-only SQLite query'}},
                               'required': ['sql'], 'additionalProperties': False}}}

    def apply(self, **kwargs):
        return ToolOutput(name=self.name, error='SQL must execute through TVCacheEnv')


class FinishSqlTool(BaseTool):
    """Submit the final SQL query as response and end the attempt."""
    def get_json_schema(self):
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description,
                'parameters': {'type': 'object', 'properties': {'response': {'type': 'string', 'description': 'Final SQL query only'}},
                               'required': ['response'], 'additionalProperties': False}}}

    def apply(self, **kwargs):
        return ToolOutput(name=self.name, error='Finish must be handled by TVCacheEnv')
