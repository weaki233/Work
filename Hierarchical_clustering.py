# -*- coding:UTF-8 -*- #
"""
@filename:Hierarchical_clustering.py
@author:Weaki
@time:2025-10-08
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time


class HierarchicalCluster:
    """
    Excel数据的多级层次聚类工具

    特点:
    - 支持任意多级分层
    - 只存储行索引,节省内存
    - 自动检测空子表格
    - 递归构建完整层次结构
    """

    def __init__(self, df: pd.DataFrame, columns: List[str], complete_mode: bool = False):
        """
        初始化聚类器

        Args:
            df: 原始DataFrame
            columns: 按顺序的分组字段列表,如['A', 'B', 'C']
            complete_mode: 是否显示所有可能的组合(包括空的)
                          False(默认): 只显示数据中存在的组合
                          True: 显示所有理论上可能的组合
        """
        self.df = df
        self.columns = columns
        self.complete_mode = complete_mode
        self.cluster_tree = {}
        self.empty_clusters = []

        # 如果是完整模式,预先获取每个字段的所有可能值
        if complete_mode:
            self.all_values = {col: sorted(df[col].unique().tolist()) for col in columns}

    def cluster(self) -> Dict[str, Any]:
        """
        执行多级聚类

        Returns:
            包含完整层次结构的字典
        """
        start_time = time.time()

        # 获取所有行索引
        all_indices = list(self.df.index)

        # 递归构建聚类树
        self.cluster_tree = self._build_cluster_tree(
            indices=all_indices,
            level=0,
            path=[]
        )

        elapsed_time = time.time() - start_time

        return {
            'tree': self.cluster_tree,
            'empty_clusters': self.empty_clusters,
            'stats': {
                'total_rows': len(self.df),
                'levels': len(self.columns),
                'empty_count': len(self.empty_clusters),
                'time_elapsed': f"{elapsed_time:.4f}秒"
            }
        }

    def _build_cluster_tree(self, indices: List[int], level: int, path: List[str]) -> Dict:
        """
        递归构建聚类树

        Args:
            indices: 当前层级的行索引列表
            level: 当前层级(0-based)
            path: 当前路径,如['A1', 'B2']

        Returns:
            当前层级的聚类字典
        """
        # 如果已经到达最后一层
        if level >= len(self.columns):
            return {}

        current_col = self.columns[level]
        cluster_dict = {}

        # 决定要处理的值列表
        if self.complete_mode:
            # 完整模式: 处理该字段的所有可能值
            values_to_process = self.all_values[current_col]
        else:
            # 精简模式: 只处理当前数据中存在的值
            if len(indices) == 0:
                return {}
            values_to_process = sorted(self.df.loc[indices, current_col].unique())

        # 按当前字段分组
        groups = defaultdict(list)
        for idx in indices:
            value = self.df.loc[idx, current_col]
            groups[value].append(idx)

        # 对每个值进行处理
        for value in values_to_process:
            group_indices = groups.get(value, [])  # 如果不存在则为空列表
            current_path = path + [str(value)]
            path_str = '->'.join(current_path)

            # 检查是否为空(完整模式下会记录空组合)
            if len(group_indices) == 0 and self.complete_mode:
                self.empty_clusters.append(path_str)

            # 构建当前节点信息
            node_info = {
                'path': path_str,
                'level': level,
                'field': current_col,
                'value': value,
                'row_indices': group_indices,
                'row_count': len(group_indices),
                'children': {},
                'is_empty': len(group_indices) == 0
            }

            # 🔄 递归构建子树 - 这是递归调用的地方！
            if level < len(self.columns) - 1:
                node_info['children'] = self._build_cluster_tree(
                    indices=group_indices,  # 传递当前组的行索引到下一层
                    level=level + 1,  # 层级+1
                    path=current_path  # 累积的路径
                )

            cluster_dict[value] = node_info

        return cluster_dict

    def get_sub_dataframe(self, path: List[Any]) -> pd.DataFrame:
        """
        根据路径获取子DataFrame

        Args:
            path: 路径列表,如['A1', 'B2', 'C1']

        Returns:
            对应的子DataFrame
        """
        node = self._navigate_to_node(path)
        if node and 'row_indices' in node:
            return self.df.loc[node['row_indices']]
        return pd.DataFrame()

    def _navigate_to_node(self, path: List[Any]) -> Dict:
        """导航到指定路径的节点"""
        current = self.cluster_tree
        for i, value in enumerate(path):
            if value not in current:
                return None
            current = current[value]
            if i < len(path) - 1:
                current = current.get('children', {})
        return current

    def print_tree(self, max_depth: int = None):
        """
        打印聚类树结构

        Args:
            max_depth: 最大显示深度,None表示显示全部
        """
        print("\n" + "=" * 60)
        print("聚类树结构")
        print("=" * 60)
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
            print(f"{indent}├─ {key} ({count}行){empty_mark} - {path}")

            if 'children' in info and info['children']:
                self._print_node(info['children'], depth + 1, max_depth)

    def get_all_leaf_nodes(self) -> List[Dict]:
        """
        获取所有叶子节点(最底层的分组)

        Returns:
            所有叶子节点的列表
        """
        leaves = []
        self._collect_leaves(self.cluster_tree, leaves)
        return leaves

    def _collect_leaves(self, node: Dict, leaves: List):
        """递归收集叶子节点"""
        for key, info in node.items():
            if not info.get('children'):
                leaves.append(info)
            else:
                self._collect_leaves(info['children'], leaves)

    def get_level_indices(self, level: int) -> Dict[str, List[int]]:
        """
        获取指定层级的所有分组及其行索引

        Args:
            level: 层级编号(0-based)，0表示第一层(A)，1表示第二层(B)，依此类推

        Returns:
            字典，key为路径字符串，value为行索引列表
            例如: {'A1': [0,1,2,5,7], 'A2': [3,4,6]}
                 或 {'A1->B1': [0,1], 'A1->B2': [2,7], ...}

        Example:
            # 获取第一层(A)的所有分组
            level_0 = clusterer.get_level_indices(0)
            # {'A1': [0,1,2,5,7], 'A2': [3,4,6]}

            # 获取第二层(B)的所有分组
            level_1 = clusterer.get_level_indices(1)
            # {'A1->B1': [0,1], 'A1->B2': [2,7], 'A1->B3': [5], ...}
        """
        if level < 0 or level >= len(self.columns):
            raise ValueError(f"层级必须在 0 到 {len(self.columns) - 1} 之间")

        result = {}
        self._collect_level_nodes(self.cluster_tree, 0, level, result)
        return result

    def _collect_level_nodes(self, node: Dict, current_level: int, target_level: int, result: Dict):
        """递归收集指定层级的节点"""
        for key, info in node.items():
            if current_level == target_level:
                # 到达目标层级，记录该节点
                result[info['path']] = info['row_indices']
            elif current_level < target_level and info.get('children'):
                # 还没到目标层级，继续递归
                self._collect_level_nodes(info['children'], current_level + 1, target_level, result)

    def get_level_dataframes(self, level: int) -> Dict[str, pd.DataFrame]:
        """
        获取指定层级的所有分组及其对应的DataFrame

        Args:
            level: 层级编号(0-based)

        Returns:
            字典，key为路径字符串，value为对应的DataFrame

        Example:
            # 获取第二层所有分组的数据表
            dfs = clusterer.get_level_dataframes(1)
            for path, df in dfs.items():
                print(f"{path}: {len(df)}行")
                print(df)
        """
        indices_dict = self.get_level_indices(level)
        return {path: self.df.loc[indices] for path, indices in indices_dict.items()}

    def print_level_summary(self, level: int):
        """
        打印指定层级的汇总信息

        Args:
            level: 层级编号(0-based)
        """
        indices_dict = self.get_level_indices(level)

        print(f"\n{'=' * 60}")
        print(f"第 {level} 层 (字段: {self.columns[level]}) 的所有分组")
        print(f"{'=' * 60}")

        for path, indices in sorted(indices_dict.items()):
            empty_mark = " ⚠️ [空]" if len(indices) == 0 else ""
            print(f"{path}: {len(indices)}行{empty_mark}")
            print(f"  行索引: {indices}")

        print(f"\n总计: {len(indices_dict)} 个分组")

    def export_summary(self) -> pd.DataFrame:
        """
        导出聚类摘要为DataFrame

        Returns:
            包含所有分组信息的摘要表
        """
        leaves = self.get_all_leaf_nodes()

        summary_data = []
        for leaf in leaves:
            row_data = {
                'path': leaf['path'],
                'row_count': leaf['row_count'],
                'level': leaf['level']
            }

            # 添加每一级的值
            path_parts = leaf['path'].split('->')
            for i, col in enumerate(self.columns[:len(path_parts)]):
                row_data[col] = path_parts[i]

            summary_data.append(row_data)

        return pd.DataFrame(summary_data)


# ============ 使用示例 ============

def example_usage():
    """完整使用示例"""

    # 1. 创建示例数据
    data = {
        'A': ['A1', 'A1', 'A1', 'A2', 'A2', 'A1', 'A2', 'A1'],
        'B': ['B1', 'B1', 'B2', 'B1', 'B2', 'B3', 'B3', 'B2'],
        'C': ['C1', 'C2', 'C1', 'C1', 'C2', 'C1', 'C1', 'C3'],
        'Value': [100, 200, 150, 300, 250, 180, 220, 160]
    }
    df = pd.DataFrame(data)

    print("原始数据:")
    print(df)
    print("\n")

    # 2. 创建聚类器 (默认模式：只显示存在的组合)
    print("=" * 60)
    print("模式1: 精简模式 (只显示数据中存在的组合)")
    print("=" * 60)
    clusterer = HierarchicalCluster(df, columns=['A', 'B', 'C'], complete_mode=False)
    result = clusterer.cluster()
    clusterer.print_tree()

    # 3. 完整模式示例
    print("\n" + "=" * 60)
    print("模式2: 完整模式 (显示所有可能的组合，包括空的)")
    print("=" * 60)
    clusterer_complete = HierarchicalCluster(df, columns=['A', 'B', 'C'], complete_mode=True)
    result_complete = clusterer_complete.cluster()
    clusterer_complete.print_tree()

    # 4. 打印统计信息
    print("\n精简模式统计:")
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")

    print("\n完整模式统计:")
    for key, value in result_complete['stats'].items():
        print(f"  {key}: {value}")

    # 5. 获取特定路径的子表格
    print("\n" + "=" * 60)
    print("示例: 获取A1->B1路径的子表格")
    print("=" * 60)
    sub_df = clusterer.get_sub_dataframe(['A1', 'B1'])
    print(sub_df)

    # 6. 演示递归过程
    print("\n" + "=" * 60)
    print("递归过程说明:")
    print("=" * 60)
    print("Level 0 (字段A): 处理 A1, A2")
    print("  → A1 递归调用 → Level 1 (字段B): 处理 B1, B2, B3")
    print("    → A1->B1 递归调用 → Level 2 (字段C): 处理 C1, C2")
    print("    → A1->B2 递归调用 → Level 2 (字段C): 处理 C1, C3")
    print("    → A1->B3 递归调用 → Level 2 (字段C): 处理 C1")
    print("  → A2 递归调用 → Level 1 (字段B): 处理 B1, B2, B3")
    print("    ... 以此类推")

    return clusterer


# ============ 从Excel文件读取的示例 ============

def example_from_excel():
    """从Excel文件读取并聚类的示例"""

    # 读取Excel文件
    df = pd.read_excel(r"C:\Users\weaki\Desktop\test.xlsx")

    # 假设Excel有字段: 'Region', 'Category', 'SubCategory', 'Sales'
    clusterer = HierarchicalCluster(df, columns=['A', 'B', 'C'])
    result = clusterer.cluster()
    clusterer.print_tree()

    # 获取特定组合的数据
    sub_df = clusterer.get_sub_dataframe(['A1', 'B1', 'C1'])
    print(sub_df)

    # 获取第1层所有分组的DataFrame字典
    level_1_dfs = clusterer.get_level_dataframes(1)

    # 遍历使用
    for path, df in level_1_dfs.items():
        print(f"{path}: {len(df)}行")
        print(df)


if __name__ == "__main__":
    # 运行示例
    # clusterer = example_usage()
    example_from_excel()
    # print("\n" + "=" * 60)
    # print("使用提示:")
    # print("=" * 60)
    # print("1. 创建聚类器: clusterer = HierarchicalCluster(df, ['A', 'B', 'C'])")
    # print("2. 执行聚类: result = clusterer.cluster()")
    # print("3. 打印树: clusterer.print_tree()")
    # print("4. 获取子表: sub_df = clusterer.get_sub_dataframe(['A1', 'B2'])")
    # print("5. 获取叶子节点: leaves = clusterer.get_all_leaf_nodes()")
    # print("6. 导出摘要: summary = clusterer.export_summary()")
