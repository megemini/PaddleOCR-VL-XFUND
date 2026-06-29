#!/usr/bin/env python3
"""
将 XFUND 数据集解析为键值对形式。

规则：
- 如果存在 question-answer 的 linking 关系，则提取为 "question_text": "answer_text"
- 如果没有 linking 关系（如 header、other），则采用通用模式 "label": "text"
"""

import json
import argparse
from collections import OrderedDict
from pathlib import Path


def parse_document(doc_items: list) -> list[OrderedDict]:
    """解析单个文档的标注项，返回键值对列表。"""
    # 构建 id -> item 映射
    id_map = {item["id"]: item for item in doc_items}

    # 收集所有被 linking 引用的 answer id，避免重复提取
    linked_answer_ids = set()
    results = []

    # 第一轮：处理有 linking 关系的 question -> answer
    for item in doc_items:
        if not item.get("linking"):
            continue

        for link in item["linking"]:
            src_id, tgt_id = link[0], link[1]

            # 只在 question 端提取，避免重复
            if item["label"] == "question" and item["id"] == src_id:
                question_item = id_map.get(src_id)
                answer_item = id_map.get(tgt_id)

                if question_item and answer_item:
                    key = question_item["text"]
                    value = answer_item["text"]
                    results.append(OrderedDict([(key, value)]))
                    linked_answer_ids.add(tgt_id)

    # 第二轮：处理没有 linking 的项（header, other, 以及未被引用的 answer）
    for item in doc_items:
        has_linking = bool(item.get("linking"))
        is_linked_answer = item["id"] in linked_answer_ids
        is_question_with_link = (
            item["label"] == "question" and has_linking and item["id"] == item["linking"][0][0]
        )

        # 跳过已经作为 question 参与键值对提取的项
        if is_question_with_link:
            continue
        # 跳过已经作为 answer 被提取的项
        if is_linked_answer:
            continue

        # 无 linking 的项，或 question 的 linking 指向的 answer 不存在的项
        if not has_linking:
            key = item["label"]
            value = item["text"]
            results.append(OrderedDict([(key, value)]))

    return results


def parse_xfund(input_path: str, output_path: str | None = None):
    """解析 XFUND JSON 文件，输出键值对。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = OrderedDict()
    all_results["lang"] = data.get("lang", "")
    all_results["split"] = data.get("split", "")
    all_results["documents"] = []

    for doc in data["documents"]:
        doc_kv = OrderedDict()
        doc_kv["id"] = doc["id"]
        doc_kv["key_value_pairs"] = parse_document(doc["document"])
        all_results["documents"].append(doc_kv)

    # 输出
    output_json = json.dumps(all_results, ensure_ascii=False, indent=2)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"结果已保存到: {output_path}")
    else:
        print(output_json)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 XFUND 数据集为键值对")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/xfund/zh.val.json",
        help="输入 XFUND JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认打印到终端）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path

    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).parent / output_path

    parse_xfund(str(input_path), str(output_path) if output_path else None)
