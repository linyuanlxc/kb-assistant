"""测试 Ragas 评估系统安装。

运行此脚本验证：
1. Ragas 依赖是否正确安装
2. 评估器是否可以正常初始化
3. 指标计算是否正常
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def test_imports():
    """测试关键模块导入。"""
    print("[TEST] 测试模块导入...")

    try:
        from core.evaluation.recorder import EvaluationRecorder
        print("  ✅ core.evaluation.recorder 导入成功")
    except Exception as e:
        print(f"  ❌ core.evaluation.recorder 导入失败: {e}")
        return False

    try:
        from core.evaluation.evaluator import RagasEvaluator
        print("  ✅ core.evaluation.evaluator 导入成功")
    except Exception as e:
        print(f"  ❌ core.evaluation.evaluator 导入失败: {e}")
        return False

    try:
        import ragas
        print(f"  ✅ ragas 版本: {ragas.__version__}")
    except Exception as e:
        print(f"  ❌ ragas 导入失败: {e}")
        return False

    return True


def test_recorder():
    """测试 EvaluationRecorder。"""
    print("\n[TEST] 测试 EvaluationRecorder...")

    from core.evaluation.recorder import EvaluationRecorder

    try:
        recorder = EvaluationRecorder(log_dir=ROOT_DIR / "runtime" / "eval")
        print("  ✅ EvaluationRecorder 初始化成功")
        print(f"  ✅ 日志文件: {recorder.get_log_file()}")
        return True
    except Exception as e:
        print(f"  ❌ EvaluationRecorder 测试失败: {e}")
        return False


def test_evaluator_initialization():
    """测试 RagasEvaluator 初始化。"""
    print("\n[TEST] 测试 RagasEvaluator 初始化...")

    try:
        from webapp.bootstrap import get_embeddings, get_registry

        registry = get_registry()
        embeddings = get_embeddings()

        llm = registry.build_chat("fast_model")
        print(f"  ✅ LLM 创建成功: {llm}")

        from core.evaluation.evaluator import RagasEvaluator

        evaluator = RagasEvaluator(llm=llm, embeddings=embeddings)
        print("  ✅ RagasEvaluator 初始化成功")
        return True
    except Exception as e:
        print(f"  ❌ RagasEvaluator 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_single_evaluation():
    """测试单条评估。"""
    print("\n[TEST] 测试单条评估...")

    try:
        from webapp.bootstrap import get_embeddings, get_registry
        from core.evaluation.evaluator import RagasEvaluator

        registry = get_registry()
        embeddings = get_embeddings()

        evaluator = RagasEvaluator(llm=registry.build_chat("fast_model"), embeddings=embeddings)

        # 模拟数据
        query = "什么是知识库助手？"
        answer = "知识库助手是一个基于 RAG 的系统，可以回答用户问题。"
        contexts = [
            "知识库助手是基于检索增强生成（RAG）技术的问答系统。",
            "它可以检索相关文档并生成答案。",
        ]

        result = evaluator.evaluate_single(query=query, answer=answer, contexts=contexts)

        if "error" in result:
            print(f"  ❌ 评估失败: {result['error']}")
            return False

        print(f"  ✅ 评估成功:")
        for metric, value in result.items():
            print(f"     {metric}: {value:.4f}")

        return True
    except Exception as e:
        print(f"  ❌ 单条评估失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("RAGAS EVALUATION SYSTEM - INSTALLATION TEST")
    print("=" * 80)

    tests = [
        ("模块导入", test_imports),
        ("EvaluationRecorder", test_recorder),
        ("RagasEvaluator 初始化", test_evaluator_initialization),
        ("单条评估", test_single_evaluation),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n[ERROR] 测试 '{name}' 异常: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 80)

    if passed == total:
        print("\n🎉 所有测试通过！评估系统已准备就绪。")
        print("\n下一步操作:")
        print("  1. 启动系统: uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload")
        print("  2. 使用系统进行问答，生成评估数据")
        print("  3. 运行评估: python scripts/eval/run_batch_eval.py")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复问题。")
        sys.exit(1)


if __name__ == "__main__":
    main()
