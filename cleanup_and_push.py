"""
清理项目并推送到GitHub
"""
import os
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.absolute()

# 需要删除的文件（测试代码和不相干的内容）
files_to_remove = [
    "test_environment.py",
    "test_simple.py", 
    "run_analysis.bat",
    "run_analysis.ps1",
    "analyze_distribution.ipynb",
    "NEXT_STEPS.md",
]

print("\n" + "="*70)
print("🧹 清理项目文件")
print("="*70 + "\n")

print("📝 需要删除的文件:")
for file in files_to_remove:
    file_path = project_root / file
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"  ✓ 删除: {file}")
        except Exception as e:
            print(f"  ✗ 失败: {file} - {e}")
    else:
        print(f"  - 不存在: {file}")

print("\n" + "="*70)
print("📦 执行Git操作")
print("="*70 + "\n")

os.chdir(project_root)

# 1. 查看status
print("1️⃣ 查看git状态...")
result = subprocess.run(["git", "status"], capture_output=True, text=True)
print(result.stdout)

# 2. 添加所有修改
print("\n2️⃣ 添加所有修改到暂存区...")
result = subprocess.run(["git", "add", "."], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✓ 操作成功")
else:
    print(f"   ✗ 出错: {result.stderr}")

# 3. 提交
print("\n3️⃣ 提交更改...")
commit_message = """改进: 添加LSTM/Transformer模型、tqdm进度条和诊断工具

核心改进:
- 新增LSTM和Transformer序列模型用于时间序列预测
- 统一的训练脚本(train_unified.py)支持多种模型对比
- 添加tqdm进度条显示训练进度和实时loss
- 改进FC模型训练(学习率、AdamW、学习率调度、早停)

诊断工具:
- 特征质量诊断脚本(diagnose_features.py)
- 数据分布分析脚本(quick_analysis.py)
- 详细的诊断报告和实验计划文档

文档:
- SOLUTION.md: 完整解决方案指南
- EXPERIMENT_PLAN.txt: 详细的实验对比计划
- DIAGNOSIS_REPORT.txt: 多维度的问题诊断分析

清理:
- 移除测试脚本和启动脚本
- 清理不相干的notebook文件"""

result = subprocess.run(["git", "commit", "-m", commit_message], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✓ 提交成功")
    print(result.stdout)
else:
    print(f"   ⚠️ 提交信息: {result.stdout}")
    if "nothing to commit" in result.stdout:
        print("   (没有新的改动需要提交)")

# 4. 推送
print("\n4️⃣ 推送到GitHub...")
result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✓ 推送成功!")
    print(result.stdout)
else:
    if "fatal" in result.stderr:
        print(f"   ✗ 错误: {result.stderr}")
    else:
        print(result.stdout)
        print(result.stderr)

print("\n" + "="*70)
print("✅ 完成!")
print("="*70 + "\n")
