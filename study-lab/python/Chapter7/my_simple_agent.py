import re
from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message, ToolRegistry

class MySimpleAgent(SimpleAgent):
    """
    Rewritten simple conversation Agent
    Demonstrates how to build a custom Agent by extending SimpleAgent
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        Rewritten run method - implements simple conversation logic, supports optional tool calling
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # Build message list
        messages = []

        # Add system message (may include tool information)
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # Add history messages
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": input_text})

        # If tool calling is not enabled, use simple conversation logic
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response

        # Logic supporting multiple rounds of tool calling
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """Build enhanced system prompt, including tool information"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # Get tool description
        tool_description = self.tool_registry.get_tools_description()
        if not tool_description or tool_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tool_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """Running logic supporting tool calling"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # Call LLM
            response = self.llm.invoke(messages, **kwargs)

            # Check if there are tool calls
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                # Execute all tool calls and collect results
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # Remove tool call markers from response
                    clean_response = clean_response.replace(call['original'], "")

                # Build message containing tool results
                messages.append({"role": "assistant", "content": clean_response})

                # Add tool results
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # No tool calls, this is the final answer
            final_response = response
            break

        # If maximum iterations exceeded, get last response
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """Parse tool calls in text"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        maches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in maches:
            tool_calls.append({
                "tool_name": tool_name.strip(),
                "parameters": parameters.strip(),
                "original": f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """Execute tool call"""
        if not self.tool_registry:
            return f"❌ Error: Tool registry not configured"

        try:
            # Intelligent parameter parsing
            if tool_name == 'calculator':
                # Calculator tool directly passes expression
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # Other tools use intelligent parameter parsing
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ Error: Tool '{tool_name}' not found"
                result = tool.run(param_dict)

            return f"🔧 Tool {tool_name} execution result:\n{result}"

        except Exception as e:
            return f"❌ Tool call failed: {str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """Intelligently parse tool parameters"""
        param_dict = {}

        if '=' in parameters:
            # Format: key=value or action=search,query=Python
            if ',' in parameters:
                # Multiple parameters: action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # Single parameter: key=value
                key, value = parameters.split('=', 1)
                param_dict[key] = value.strip()
        else:
            # Directly pass parameters, intelligently infer based on tool type
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input', parameters}

        return param_dict

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        Custom streaming run method
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # Stream call LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        print()  # New line

        # Save complete conversation to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """Add tool to Agent (convenience method)"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """Check if tools are available"""
        return self.enable_tool_calling and self.tool_registry

    def remove_tool(self, tool_name: str) -> bool:
        """Remove tool (convenience method)"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False

    def list_tools(self) -> list:
        """List all available tools"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
