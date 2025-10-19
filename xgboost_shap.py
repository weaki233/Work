# -*- coding: utf-8 -*-

# 导入所有需要的库
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gooey import Gooey, GooeyParser
import sys
import io
# --- 强制标准输出/错误流使用 UTF-8 编码并启用行缓冲 ---
# 这是一个处理打包后程序（尤其是在Windows上）Unicode错误和输出延迟问题的稳定方法。
# line_buffering=True 确保每行 print 输出后都会立即刷新，实现实时显示。
if sys.stdout is not None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
if sys.stderr is not None:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
# --- 修复结束 ---
# --- 核心分析逻辑 (修复并改进版) ---
def perform_analysis(excel_path, target_col, feature_cols_str):
    """
    执行数据加载、模型训练、SHAP分析和可视化的主函数。
    """
    try:
        print(f"--- 1. 正在读取 Excel 文件: {excel_path} ---")
        df = pd.read_excel(excel_path)
        # 预先删除目标列本身为空的行，避免模型训练出错
        df.dropna(subset=[target_col], inplace=True)
        print("文件读取成功！\n")
    except Exception as e:
        print(f"错误：无法读取 Excel 文件。请检查文件路径和格式是否正确。\n{e}")
        return

    # --- 2. 数据准备与验证 ---
    print("--- 2. 正在准备数据 ---")
    if target_col not in df.columns:
        print(f"错误：目标列 '{target_col}' 在文件中不存在！")
        print(f"文件中的所有列为: {df.columns.tolist()}")
        return

    feature_cols = [col.strip() for col in feature_cols_str.split(',') if col.strip()]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：以下特征列在文件中不存在: {missing_cols}")
        return

    print(f"目标 (Y): {target_col}")
    print(f"选择的特征 (X): {feature_cols}\n")

    X = df[feature_cols]
    y = df[target_col]

    # --- 3. 自动处理非数值特征与缺失值 ---
    print("--- 3. 正在对特征进行预处理 ---")
    # 使用独热编码，将非数值 (分类) 特征转换为模型可用的数值特征
    X_processed = pd.get_dummies(X, dummy_na=False, drop_first=True)

    if X_processed.shape[1] > X.shape[1]:
        print("注意：已自动对非数值特征进行了独热编码。")

    # 对处理后数据中可能存在的数值型缺失值 (NaN) 用中位数进行填充
    if X_processed.isnull().sum().sum() > 0:
        print("注意：发现数据中存在缺失值，已自动使用中位数进行填充。")
        X_processed.fillna(X_processed.median(), inplace=True)

    if X_processed.empty:
        print("错误：处理后没有可用的特征进行分析。")
        return

    print(f"最终用于分析的特征数量: {len(X_processed.columns)}\n")

    # --- 4. 划分数据与训练模型 ---
    print("--- 4. 正在训练 XGBoost 模型 ---")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("模型训练完成！\n")

    # --- 5. 使用 SHAP 进行解释 ---
    print("--- 5. 正在计算 SHAP 值 (这可能需要一些时间) ---")
    explainer = shap.TreeExplainer(model)
    # 在测试集上计算 SHAP 值，以评估模型对新数据的解释能力
    shap_values = explainer.shap_values(X_test)
    print("SHAP 值计算完成！\n")

    # --- 6. 可视化与解读结果 ---
    print("--- 6. 正在生成 SHAP 可视化图表 ---")

    # 设置 matplotlib 支持中文显示的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # # --- 【绘图逻辑 - 原版分开绘图的代码】 ---
    # # 图1：全局特征重要性条形图
    # print("生成图1: 全局特征重要性 (条形图)...")
    # shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    # fig1 = plt.gcf()
    # fig1.set_size_inches(10, 8)
    # plt.title("全局特征重要性 (条形图)", fontsize=16)
    # plt.tight_layout()
    #
    # # 图2：SHAP 摘要图 (蜂群图)
    # print("生成图2: SHAP 特征影响详解 (蜂群图)...")
    # shap.summary_plot(shap_values, X_test, show=False)
    # fig2 = plt.gcf()
    # fig2.set_size_inches(10, 8)
    # plt.title("SHAP 特征影响详解 (蜂群图)", fontsize=16)
    # plt.tight_layout()
    # # --- 【原版代码结束】 ---

    # --- 【绘图逻辑 - 合并版】 ---
    print("正在生成合并版 SHAP 摘要图...")

    # 1. 创建一个画布和第一个坐标轴 (ax1)，用于绘制蜂群图
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title("SHAP 特征影响分析 (蜂群图 & 特征重要性)", fontsize=16, pad=40)

    # 2. 在 ax1 上绘制蜂群图 (dot plot)
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, color_bar=True)

    # 3. 创建一个共享 Y 轴的第二个坐标轴 (ax2)，用于绘制条形图
    ax2 = ax1.twiny()

    # 4. 在 ax2 上绘制条形图 (bar plot)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    # 5. 美化与调整
    # a. 将条形图的柱子设为半透明，以免完全遮挡后面的蜂群图
    for bar in ax2.patches:
        bar.set_alpha(0.2)

    # b. 为两个坐标轴设置清晰的标签
    ax1.set_xlabel("SHAP 值 (对单个样本预测的影响)", fontsize=12)
    ax2.set_xlabel("平均 |SHAP 值| (全局特征重要性)", fontsize=12)

    # c. 将第二个坐标轴 (ax2) 的标签和刻度移动到图表顶部
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    # d. 调整布局，防止标签重叠
    fig.tight_layout(pad=1.5)

    # --- 【合并版代码结束】 ---

    # --- 7. 生成部分最重要特征的依赖图 ---
    print("\n--- 7. 正在生成特征依赖图 (每个图会单独弹出) ---")

    # 创建一个与 X_test 对应的原始特征 DataFrame，用于依赖图的智能着色
    X_test_for_display = X.loc[X_test.index]

    # 计算全局特征重要性，并获取最重要的特征
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_processed.columns, mean_abs_shap)), columns=['特征', '重要性'])
    feature_importance = feature_importance.sort_values('重要性', ascending=False)

    # 确定要绘制的图的数量（最多5个，或总特征数，取较小者）
    num_plots_to_show = min(5, len(X_processed.columns))
    top_features = feature_importance['特征'].tolist()[:num_plots_to_show]

    print(f"将为最重要的 {num_plots_to_show} 个特征生成依赖图: {top_features}")

    for feature_name in top_features:
        # 同样，让 shap.dependence_plot 创建图形，然后我们再调整
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_test,
            display_features=X_test_for_display,
            show=False
        )
        fig_dep = plt.gcf()
        fig_dep.set_size_inches(7, 5)
        plt.title(f"特征 '{feature_name}' 的依赖图", fontsize=14)
        plt.tight_layout()

    # 所有图表准备好后，一次性显示出来
    print("\n所有图表已生成，请查看弹出的窗口。")
    print("关闭所有图表窗口后，程序将退出。")
    plt.show()

    print("\n--- 所有分析已完成！---")


# --- Gooey 主程序 ---
@Gooey(
    program_name="SHAP 特征重要性分析工具",
    program_description="上传 Excel 文件，自动训练模型并分析各因素对目标的影响",
    language='chinese',
    default_size=(720, 680),
    progress_regex=r"^--- (\d+)\. .*",
    progress_expr="x[0]"
)
def main():
    parser = GooeyParser(description="请提供您的数据文件和相关列名")

    parser.add_argument(
        'excel_file',
        metavar='Excel 文件',
        help="请选择包含您数据的 Excel 文件 (.xlsx 或 .xls)",
        widget='FileChooser',
        gooey_options={
            'wildcard': "Excel files (*.xlsx, *.xls)|*.xlsx;*.xls",
            'message': "选择一个文件"
        }
    )

    parser.add_argument(
        'target_column',
        metavar='目标列名',
        help="您想要预测的列的名称 (例如: salary, sales, score)",
        default='income_level'
    )

    parser.add_argument(
        'feature_columns',
        metavar='相关因素列表 (特征列)',
        help="输入所有可能相关的列名，并用英文逗号 (,) 隔开\n例如: experience,education_level,performance_score",
        default='Age,Workclass,Education-Num,Marital Status,Occupation,Relationship,Race,Sex,Capital Gain,Capital Loss,Hours per week,Country',
    )

    args = parser.parse_args()

    perform_analysis(args.excel_file, args.target_column, args.feature_columns)


if __name__ == '__main__':
    main()
