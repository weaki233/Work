# -*- coding:UTF-8 -*- #
"""
@filename:Hierarchical_clustering_export_final_with_print.py
@author:Weaki
@time:2025-10-08
"""

import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
import time
# 导入用于Excel格式化的库
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter


class HierarchicalCluster:
    """
    Excel数据的多级层次聚类工具

    特点:
    - 支持任意多级分层
    - 只存储行索引,节省内存
    - 自动检测空子表格
    - 递归构建完整层次结构
    - [最终版] 支持按层级导出到Excel，实现正确排序、单元格合并、添加计数和居中格式化
    """

    def __init__(self, df: pd.DataFrame, columns: List[str], complete_mode: bool = False):
        self.df = df
        self.columns = columns
        self.complete_mode = complete_mode
        self.cluster_tree = {}
        self.empty_clusters = []
        if complete_mode:
            self.all_values = {col: sorted(df[col].unique().tolist()) for col in columns}

    def cluster(self) -> Dict[str, Any]:
        start_time = time.time()
        all_indices = list(self.df.index)
        self.cluster_tree = self._build_cluster_tree(indices=all_indices, level=0, path=[])
        elapsed_time = time.time() - start_time
        return {
            'tree': self.cluster_tree, 'empty_clusters': self.empty_clusters,
            'stats': {
                'total_rows': len(self.df), 'levels': len(self.columns),
                'empty_count': len(self.empty_clusters), 'time_elapsed': f"{elapsed_time:.4f}秒"
            }
        }

    def _build_cluster_tree(self, indices: List[int], level: int, path: List[str]) -> Dict:
        if level >= len(self.columns): return {}
        current_col = self.columns[level]
        cluster_dict = {}
        if self.complete_mode:
            values_to_process = self.all_values[current_col]
        else:
            if len(indices) == 0: return {}
            values_to_process = sorted(self.df.loc[indices, current_col].unique())
        groups = defaultdict(list)
        for idx in indices:
            value = self.df.loc[idx, current_col]
            groups[value].append(idx)
        for value in values_to_process:
            group_indices = groups.get(value, [])
            current_path = path + [str(value)]
            path_str = '->'.join(current_path)
            if len(group_indices) == 0 and self.complete_mode: self.empty_clusters.append(path_str)
            node_info = {
                'path': path_str, 'level': level, 'field': current_col, 'value': value,
                'row_indices': group_indices, 'row_count': len(group_indices),
                'children': {}, 'is_empty': len(group_indices) == 0
            }
            if level < len(self.columns) - 1:
                node_info['children'] = self._build_cluster_tree(indices=group_indices, level=level + 1,
                                                                 path=current_path)
            cluster_dict[value] = node_info
        return cluster_dict

    # =================================================================
    # 🌟 新增：打印树结构的方法
    # =================================================================
    def print_tree(self, max_depth: int = None):
        """
        打印聚类树结构

        Args:
            max_depth: 最大显示深度,None表示显示全部
        """
        print("\n" + "=" * 60)
        print("🌳 聚类树结构")
        print("=" * 60)
        if not self.cluster_tree:
            print("树为空，请先运行 .cluster() 方法。")
            return

        self._print_node(self.cluster_tree, 0, max_depth)

        if self.empty_clusters:
            print("\n" + "=" * 60)
            print(f"空子表格警告 (共{len(self.empty_clusters)}个):")
            print("=" * 60)
            for path in self.empty_clusters:
                print(f"  ⚠ {path}")

    def _print_node(self, node: Dict, depth: int, max_depth: int):
        """递归打印节点"""
        if max_depth is not None and depth >= max_depth:
            return

        for key, info in sorted(node.items()):
            indent = "  " * depth
            path = info.get('path', str(key))
            count = info.get('row_count', 0)
            is_empty = info.get('is_empty', False)

            # 空节点用特殊标记
            empty_mark = " ⚠️ [空]" if is_empty else ""
            print(f"{indent}├─ {key} ({count}行){empty_mark}")

            if 'children' in info and info['children']:
                self._print_node(info['children'], depth + 1, max_depth)

    def get_level_indices(self, level: int) -> Dict[str, List[int]]:
        if level < 0 or level >= len(self.columns):
            raise ValueError(f"层级必须在 0 到 {len(self.columns) - 1} 之间")
        result = {};
        self._collect_level_nodes(self.cluster_tree, 0, level, result);
        return result

    def _collect_level_nodes(self, node: Dict, current_level: int, target_level: int, result: Dict):
        for key, info in node.items():
            if current_level == target_level:
                result[info['path']] = info['row_indices']
            elif current_level < target_level and info.get('children'):
                self._collect_level_nodes(info['children'], current_level + 1, target_level, result)

    def _format_and_merge_sheet(self, worksheet, merge_cols_indices: List[int]):
        """
        对给定的worksheet进行格式化：居中、合并、调整列宽
        """
        if not worksheet:
            return

        center_align = Alignment(horizontal='center', vertical='center')

        # 1. 居中所有单元格
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = center_align

        # 2. 合并指定列的单元格
        for col_idx in merge_cols_indices:
            start_row = 2
            for i in range(2, worksheet.max_row + 2):
                is_last_row = (i == worksheet.max_row + 1)
                current_cell_value = worksheet.cell(row=i, column=col_idx + 1).value
                start_cell_value = worksheet.cell(row=start_row, column=col_idx + 1).value

                if is_last_row or (current_cell_value != start_cell_value):
                    if i - 1 > start_row:
                        worksheet.merge_cells(start_row=start_row, start_column=col_idx + 1,
                                              end_row=i - 1, end_column=col_idx + 1)
                    start_row = i

        # 3. 自动调整列宽
        for col in worksheet.columns:
            max_length = 0
            column_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    # 将表头长度也纳入计算
                    header_len = len(str(worksheet.cell(row=1, column=cell.column).value))
                    cell_len = len(str(cell.value))
                    current_max = max(header_len, cell_len)
                    if current_max > max_length:
                        max_length = current_max
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            worksheet.column_dimensions[column_letter].width = adjusted_width

    def export_to_excel_by_level(self, output_path: str):
        if not self.cluster_tree:
            print("❌ 错误: 请先调用 .cluster() 方法执行聚类。")
            return
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                all_path_counts = {}

                # --- Sheet 1: 第一级 ---
                if len(self.columns) >= 1:
                    level_0_indices = self.get_level_indices(0)
                    level_0_data = []
                    for path, indices in level_0_indices.items():
                        count = len(indices);
                        all_path_counts[path] = count
                        level_0_data.append({self.columns[0]: path, '数量': count})
                    if level_0_data:
                        df_level_0 = pd.DataFrame(level_0_data).sort_values(by=['数量', self.columns[0]],
                                                                            ascending=[False, True])
                        df_level_0.to_excel(writer, sheet_name='第一级汇总', index=False)
                        self._format_and_merge_sheet(writer.sheets.get('第一级汇总'), merge_cols_indices=[])

                # --- Sheet 2: 第二级 ---
                if len(self.columns) >= 2:
                    level_1_indices = self.get_level_indices(1)
                    level_1_data = []
                    for path, indices in level_1_indices.items():
                        count = len(indices);
                        all_path_counts[path] = count
                        parts = path.split('->');
                        parent_path = parts[0]
                        level_1_data.append({
                            self.columns[0]: parts[0], self.columns[1]: parts[1], '数量': count,
                            '__parent_count': all_path_counts.get(parent_path, 0)
                        })
                    if level_1_data:
                        df_level_1 = pd.DataFrame(level_1_data).sort_values(
                            by=['__parent_count', self.columns[0], '数量'], ascending=[False, True, False]
                        )
                        df_level_1[self.columns[0]] = df_level_1.apply(
                            lambda r: f"{r[self.columns[0]]} ({r['__parent_count']})", axis=1)
                        df_level_1.drop(columns=['__parent_count']).to_excel(writer, sheet_name='第二级汇总',
                                                                             index=False)
                        self._format_and_merge_sheet(writer.sheets.get('第二级汇总'), merge_cols_indices=[0])

                # --- Sheet 3: 第三级 ---
                if len(self.columns) >= 3:
                    level_2_indices = self.get_level_indices(2)
                    level_2_data = []
                    for path, indices in level_2_indices.items():
                        count = len(indices)
                        parts = path.split('->');
                        parent_path_l1 = parts[0];
                        parent_path_l2 = '->'.join(parts[:2])
                        level_2_data.append({
                            self.columns[0]: parts[0], self.columns[1]: parts[1], self.columns[2]: parts[2],
                            '数量': count,
                            '__parent_count_l1': all_path_counts.get(parent_path_l1, 0),
                            '__parent_count_l2': all_path_counts.get(parent_path_l2, 0)
                        })
                    if level_2_data:
                        df_level_2 = pd.DataFrame(level_2_data).sort_values(
                            by=['__parent_count_l1', self.columns[0], '__parent_count_l2', self.columns[1], '数量'],
                            ascending=[False, True, False, True, False]
                        )
                        df_level_2[self.columns[0]] = df_level_2.apply(
                            lambda r: f"{r[self.columns[0]]} ({r['__parent_count_l1']})", axis=1)
                        df_level_2[self.columns[1]] = df_level_2.apply(
                            lambda r: f"{r[self.columns[1]]} ({r['__parent_count_l2']})", axis=1)
                        df_level_2.drop(columns=['__parent_count_l1', '__parent_count_l2']).to_excel(writer,
                                                                                                     sheet_name='第三级汇总',
                                                                                                     index=False)
                        self._format_and_merge_sheet(writer.sheets.get('第三级汇总'), merge_cols_indices=[0, 1])

            print(f"\n✅ 成功导出最终版格式化Excel文件到: {output_path}")
        except Exception as e:
            print(f"\n❌ 导出Excel失败: {e}")
            print("  请确保您已安装 'openpyxl' 库 (在终端或命令提示符中运行: pip install openpyxl)")


# ============ 使用示例 ============

def example_from_excel(file_path: str, columns: List[str]):
    """从Excel文件读取并聚类的示例"""
    try:
        df = pd.read_excel(file_path)
        print(f"成功从 {file_path} 读取 {len(df)} 行数据。")

        clusterer = HierarchicalCluster(df, columns=columns)
        clusterer.cluster()

        # 🌟 在这里调用打印树的方法
        clusterer.print_tree()

        # 导出到Excel
        output_filename = "Excel聚类结果(最终版).xlsx"
        clusterer.export_to_excel_by_level(output_filename)

    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 - {file_path}")
    except Exception as e:
        print(f"❌ 读取或处理Excel时发生错误: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 30, "运行Excel文件示例", "=" * 30)
    # ⚠️ 请确保您的桌面上有这个 test.xlsx 文件，或者修改为您的正确路径
    excel_file = r"C:\Users\weaki\Desktop\test.xlsx"
    # ⚠️ 请修改为您的分组列名
    group_columns = ['A', 'B', 'C']
    example_from_excel(excel_file, group_columns)