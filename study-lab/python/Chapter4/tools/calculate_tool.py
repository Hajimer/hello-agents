import ast
import math


def _calculate_node(node):
    """递归计算白名单中的 AST 节点，其他语法一律拒绝。"""
    # 数字是递归的终点。精确检查类型，避免将 bool 当作 int。
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = node.value
    elif isinstance(node, ast.BinOp):
        # 先计算左右子树；括号和优先级已经体现在树的结构中。
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        else:
            raise ValueError("仅支持 +、-、*、/ 运算。")
    elif isinstance(node, ast.UnaryOp):
        # -3 中的负号属于一元运算，与 1 - 3 中的减法不同。
        operand = _calculate_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            result = +operand
        elif isinstance(node.op, ast.USub):
            result = -operand
        else:
            raise ValueError("仅支持一元正号和负号。")
    else:
        raise ValueError("仅支持数字、四则运算和括号，不支持变量或函数调用。")

    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("数值超出有限浮点数范围。")
    return result

def calculate(query: str) -> str:
    """
    计算器工具
    计算字符串表达式，支持整数、浮点数、+、-、*、/、括号和正负号。
    例如：(123 + 456) * 789 / 12。输入不包含 = 或 ?。
    """
    print(f"🔍 正在执行计算: {query}")
    try:
        if not isinstance(query, str):
            raise ValueError("输入必须是字符串。")
        expression = query.strip()
        if not expression:
            raise ValueError("表达式不能为空。")
        if len(expression) > 500:
            raise ValueError("表达式不能超过 500 个字符。")

        # mode="eval" 表示解析单个表达式，不代表执行 eval()。
        tree = ast.parse(expression, mode="eval")
        return str(_calculate_node(tree.body))
    except ZeroDivisionError:
        return "计算时发生错误: 除数不能为 0。"
    except SyntaxError:
        return "计算时发生错误: 表达式语法无效，请使用 * 和 / 表示乘除。"
    except Exception as e:
        return f"计算时发生错误: {e}"
