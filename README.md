# PaddleOCR-VL-XFUND：微调 XFUND 与信息抽取

---

![](https://ai-studio-static-online.cdn.bcebos.com/510b3dab08f14c1fa7219ed5895e925a8f99404048154b8cb7d27f359ffefa4a)

---

| Image | Prompt | Output |
| - | - | - |
| ![](https://ai-studio-static-online.cdn.bcebos.com/cd3bd4d3e668490d8bc58575296cc8856a302cdfc4ed427c937cc3dd3f11bf64) | ![](https://ai-studio-static-online.cdn.bcebos.com/f199eb4dce934602a38345cf43e87c1e78e6ce715fc748d5baab10b76dfd53ea) | ![](https://ai-studio-static-online.cdn.bcebos.com/f7153a546b9d4b69ad37eeee8b56bc79e1a380666d7445fca6568f0762460b09) |


**项目**：
https://aistudio.baidu.com/projectdetail/10253559

**模型库**：

https://aistudio.baidu.com/modelsdetail/46994/intro

https://modelscope.cn/models/megemini/PaddleOCR-VL-XFUND/summary

**数据集**：

https://aistudio.baidu.com/dataset/detail/382107/intro

https://modelscope.cn/datasets/megemini/PaddleOCR-VL-XFUND-Dataset

---

## ✨️ 引言

去年(2025年)年底的时候，我在 [微调 PaddleOCR-VL 新姿势 -- Prompt 与 信息抽取](https://aistudio.baidu.com/projectdetail/9857242) 这篇文章中尝试了微调 PaddleOCR-VL 进行单据的信息抽取（[megemini/PaddleOCR-VL-Receipt](https://aistudio.baidu.com/modelsdetail/41446/intro)）。

实验证明：**可以将 PaddleOCR-VL 系列模型当作微型的 VLM 进行图片的关键信息抽取（KIE）。**

使用 VLM 进行 KIE 的优势是什么呢？借用《飞桨产业实践范例库》中的 [表单识别](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4002959)

![](https://ai-studio-static-online.cdn.bcebos.com/705e2523ea50470793a4ff6383ab36b5c1d7c37b7ded4c7fbd45e50df6b71f90)

传统的图片关键信息抽取任务的主要问题是，**流程长**，基本上可以认为是 `OCR引擎 + SER + RE的串联` 的流水线任务，并且，每一步都需要对模型进行单独训练。而 VLM 模型可以端到端的实现信息抽取。

当时没有 VLM 的 OCR 模型支持 KIE 任务。（注：HunyuanOCR 发布的时间与 PaddleOCR-VL-Receipt 基本是差不多的）

使用的数据是 [WildReceipt 户外票据采集数据集---关键信息提取](https://aistudio.baidu.com/dataset/detail/129038/intro) 

此次任务使用更具产业价值的 [XFUND: A Multilingual Form Understanding Benchmark](https://github.com/doc-analysis/XFUND) 数据集进行模型的微调，

此次微调的基础模型为 [megemini/PaddleOCR-VL-Receipt](https://aistudio.baidu.com/modelsdetail/41446/intro)，模型的继承关系为（下文使用 PaddleOCR-VL-XFUND 指代此次的微调的模型）：

`PaddleOCR-VL-XFUND` <- `PaddleOCR-VL-Receipt` <- `PaddleOCR-VL`

并与近期新出的也支持 KIE 的 OCR 模型：GLM-OCR 与 HunyuanOCR 进行模型性能的对比。

结论：

- PaddleOCR-VL-XFUND 以 0.4523 的准确度排名第一，GLM-OCR 0.4159 第二 。
- PaddleOCR-VL-XFUND 多语种表现优异
- PaddleOCR-VL-XFUND 的基础模型 PaddleOCR-VL-Receipt 同样具备一定的 KIE 泛化能力

> **注意**：PaddleOCR-VL-XFUND 的微调数据集为 XFUND 的 train 的部分，使用 XFUND 的 val 的部分进行评测。
>
> 由于无法确认 GLM-OCR 与 HunyuanOCR 的训练数据集中是否有 XFUND 的 val 部分，因此，此处的评测分数仅供参考。

---


## ✨️ 数据准备

XFUND 标注的数据不是针对 VLM 的格式的，下面是 XFUND 的示例：

```json
{
    "height": 3508, # 图像高度
    "width": 2480,  # 图像宽度
    "ocr_info": [
        {
            "text": "邮政地址:",  # 单个文本内容
            "label": "question", # 文本所属类别a
            "bbox": [261, 802, 483, 859], # 单个文本框
            "id": 54,  # 文本索引
            "linking": [[54, 60]], # 当前文本和其他文本的关系 [question, answer]
            "words": []
        },
        {
            "text": "湖南省怀化市市辖区",
            "label": "answer",
            "bbox": [487, 810, 862, 859],
            "id": 60,
            "linking": [[54, 60]],
            "words": []
        }
    ]
}
```

这样的格式对于 VLM 来说，首先，存在很多的冗余信息，比如坐标、bbox 等（特殊应用除外），另外，信息之间的关系也是断开的。
  
以下列图片中的 `声明` 部分来说
 
| 示例 |
| - |
| ![](https://ai-studio-static-online.cdn.bcebos.com/593798921aee4c78916f991852f58025cd44cfa3a2c74067a3130e551c793fdd) |


标注中的信息通过 `linking` 字段构建关系，整理后为

```json
{
  "声明:": "我保证,以上所提供的情况全部真实无误,绝无欺瞒,必要时,公司有权向我前任公司对我的工作情况进行"
},
{
  "声明:": "核实;若有不实,因造假、欺瞒不报而导致的劳动关系解除,责任由本人自行承担,愿接受无条件解聘的处分。"
},
```

对于 VLM 来说，真正需要的是一段连续的文本。比如：

```json
{
  "声明:": "我保证,以上所提供的情况全部真实无误,绝无欺瞒,必要时,公司有权向我前任公司对我的工作情况进行核实;若有不实,因造假、欺瞒不报而导致的劳动关系解除,责任由本人自行承担,愿接受无条件解聘的处分。"
},
```

[PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 中有对 PaddleOCR-VL 进行微调任务的数据格式要求：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

要想得到这样的数据，需要经过三步：

1. 通过 XFUND 中的 link 字段构建 KV pair
2. 使用大模型将 KV pair 整合为有意义的 json 数据
3. 构建 ERNIEKit 的 SFT VL Dataset Format

---

### 1. 构建 KV Pair

XFUND 中通过 `linking` 字段将各个分散的标注数据连接起来，比如：

```json
{
    "box": [
        228,
        2893,
        317,
        2935
    ],
    "text": "声明:",
    "label": "question",
    "words": [
        {
            "box": [
                232,
                2893,
                261,
                2933
            ],
            "text": "声"
        },
        ...
    ],
    "linking": [
        [
            498,
            502
        ],
        [
            498,
            550
        ]
    ],
    "id": 498
},
{
    "box": [
        306,
        2976,
        2028,
        3031
    ],
    "text": "我保证,以上所提供的情况全部真实无误,绝无欺瞒,必要时,公司有权向我前任公司对我的工作情况进行",
    "label": "answer",
    "words": [
        {
            "box": [
                308,
                2977,
                340,
                3022
            ],
            "text": "我"
        },
        ...
    ],
    "linking": [
        [
            498,
            502
        ]
    ],
    "id": 502
},
{
    "box": [
        228,
        3055,
        2001,
        3116
    ],
    "text": "核实;若有不实,因造假、欺瞒不报而导致的劳动关系解除,责任由本人自行承担,愿接受无条件解聘的处分。",
    "label": "answer",
    "words": [
        {
            "box": [
                231,
                3055,
                260,
                3106
            ],
            "text": "核"
        },
        ...
    ],
    "linking": [
        [
            498,
            550
        ]
    ],
    "id": 550
},
```

其中，`"声明:"` `"id": 498` 是 `question` ，`"我保证,以上所提供的情况全部真实无误,绝无欺瞒,必要时,公司有权向我前任公司对我的工作情况进行"` `"id": 502` 是 `answer`，

`498` 共有 `2` 个 `linking` 信息，因此可以得到：

```json
{
  "声明:": "我保证,以上所提供的情况全部真实无误,绝无欺瞒,必要时,公司有权向我前任公司对我的工作情况进行"
},
{
  "声明:": "核实;若有不实,因造假、欺瞒不报而导致的劳动关系解除,责任由本人自行承担,愿接受无条件解聘的处分。"
},
```

通过这样的初步处理，就可以得到一份 KV Pair。

### 2. 整合 JSON

这是数据集准备中最关键的一步。

虽然可以通过 XFUND 得到 KV Pair，但是，如果希望能够喂给 VLM 模型，需要的是完整的 JSON 格式的数据，基本要求：

- 不能有重复的 KEY
- 要有合理的层级结构

由于没有适合的算法直接实现这样的数据处理，这里通过使用 `ernie-4.5-turbo-vl` 这个模型帮我们进行数据的整理。

由于 `ernie-4.5-turbo-vl` 支持视觉，因此，我们将原始图片与 KV Pair 一并喂给模型。

这里使用的 prompt 如下：

```python
    prompt = f"""请从以上图片中抽取键值对，注意，不要有图片中不存在的信息，层级关系尽量保持简单。

参考的键值对信息如下：
{kv_text}

请根据图片内容验证并提取准确的键值对信息。
注意：
1. 只返回json格式的数据
2. 不要遗漏参考的键值对的`键`，可以进行有效的值的合并操作
"""

```

下面是经过脚本处理后的数据格式：

```json
{
    "document_id": "zh_val_8",
    "extracted_data": {
        "应聘职位第一选择": "会计",
        "性别": "男",
        "年龄": 25,
        "婚姻状况": "未婚",
        "政治面貌": "群众",
        "民族": "汉",
        "身高": 178,
        "最高学历": "本科",
        "计算机水平": "二级",
        "驾龄": "2年(ABC)驾照",
        "外语能力": {
            "语种": "英语",
            "程度": "四级"
        },
        "身份证号码": "320106199509245113",
        "户口所在地": "上海市茂龄别墅",
        "现居地址": "上海市茂龄别墅",
        "手机": "13422280044",
        "职称": "初级会计师",
        "职称专业": "会计",
        "获此职称时间": "2015.6",
        "是否有熟悉或认识的人在本公司工作": "是",
        "关系": "20181115",
        "紧急联络人姓名": "李明露",
        "紧急联络人地址": "上海市建国西路619弄小区",
        "紧急联络人电话": "15885867381",
        "目前薪酬": 7000,
        "要求最低薪酬标准": 6500,
        "如聘用,可上岗日期": "2020年7月1日",
        "教育及培训状况": [{
            "文化程度": "高中",
            "起止日期": "201009-201306",
            "院校/培训单位名称": "上海师大附中",
            "专业/培训内容": "",
            "学位/证书": ""
        }, {
            "文化程度": "本科",
            "起止日期": "201309-201706",
            "院校/培训单位名称": "上海财经大学",
            "专业/培训内容": "会计",
            "学位/证书": "学士学位"
        }],
        "家庭成员情况": [{
            "姓名": "吕大仁",
            "与本人关系": "父亲",
            "工作单位": "",
            "职务": ""
        }, {
            "姓名": "刘艳",
            "与本人关系": "母亲",
            "工作单位": "",
            "职务": ""
        }],
        "声明": "我保证，以上所提供的情况全部真实无误，绝无欺瞒，必要时，公司有权向我前任公司对我的工作情况进行核实；若有不实，因造假、欺瞒不报而导致的劳动关系解除，责任由本人自行承担，愿接受无条件解聘的处分。",
        "签名": "吕书同",
        "日期": "2019年5月8日",
        "姓名": "吕书同",
        "header": "应聘人员登记表",
        "other": "照片,电话,第二选择:,工作经历(请您对公司提供相关的证明人或推荐人,以便必要时查询,我们会设法为您保密。),起止日期,年/月—年/月,工作单位及部门,职务,薪酬,离职原因,证明人及电话"
    },
    "raw_response": "```json\n{\n  \"应聘职位第一选择\": \"会计\",\n  \"性别\": \"男\",\n  \"年龄\": 25,\n  \"婚姻状况\": \"未婚\",\n  \"政治面貌\": \"群众\",\n  \"民族\": \"汉\",\n  \"身高\": 178,\n  \"最高学历\": \"本科\",\n  \"计算机水平\": \"二级\",\n  \"驾龄\": \"2年(ABC)驾照\",\n  \"外语能力\": {\n    \"语种\": \"英语\",\n    \"程度\": \"四级\"\n  },\n  \"身份证号码\": \"320106199509245113\",\n  \"户口所在地\": \"上海市茂龄别墅\",\n  \"现居地址\": \"上海市茂龄别墅\",\n  \"手机\": \"13422280044\",\n  \"职称\": \"初级会计师\",\n  \"职称专业\": \"会计\",\n  \"获此职称时间\": \"2015.6\",\n  \"是否有熟悉或认识的人在本公司工作\": \"是\",\n  \"关系\": \"20181115\",\n  \"紧急联络人姓名\": \"李明露\",\n  \"紧急联络人地址\": \"上海市建国西路619弄小区\",\n  \"紧急联络人电话\": \"15885867381\",\n  \"目前薪酬\": 7000,\n  \"要求最低薪酬标准\": 6500,\n  \"如聘用,可上岗日期\": \"2020年7月1日\",\n  \"教育及培训状况\": [\n    {\n      \"文化程度\": \"高中\",\n      \"起止日期\": \"201009-201306\",\n      \"院校/培训单位名称\": \"上海师大附中\",\n      \"专业/培训内容\": \"\",\n      \"学位/证书\": \"\"\n    },\n    {\n      \"文化程度\": \"本科\",\n      \"起止日期\": \"201309-201706\",\n      \"院校/培训单位名称\": \"上海财经大学\",\n      \"专业/培训内容\": \"会计\",\n      \"学位/证书\": \"学士学位\"\n    }\n  ],\n  \"家庭成员情况\": [\n    {\n      \"姓名\": \"吕大仁\",\n      \"与本人关系\": \"父亲\",\n      \"工作单位\": \"\",\n      \"职务\": \"\"\n    },\n    {\n      \"姓名\": \"刘艳\",\n      \"与本人关系\": \"母亲\",\n      \"工作单位\": \"\",\n      \"职务\": \"\"\n    }\n  ],\n  \"声明\": \"我保证，以上所提供的情况全部真实无误，绝无欺瞒，必要时，公司有权向我前任公司对我的工作情况进行核实；若有不实，因造假、欺瞒不报而导致的劳动关系解除，责任由本人自行承担，愿接受无条件解聘的处分。\",\n  \"签名\": \"吕书同\",\n  \"日期\": \"2019年5月8日\",\n  \"姓名\": \"吕书同\",\n  \"header\": \"应聘人员登记表\",\n  \"other\": \"照片,电话,第二选择:,工作经历(请您对公司提供相关的证明人或推荐人,以便必要时查询,我们会设法为您保密。),起止日期,年/月—年/月,工作单位及部门,职务,薪酬,离职原因,证明人及电话\"\n}\n```",
    "status": "success",
    "fallback_used": false
}
```

其中 `extracted_data` 就是我们需要的 json 格式数据。

### 3. 构建 SFT VL Dataset


[PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 中有对微调数据格式的要求：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

其中，

- `tag` 为 `mask` 的 `text` 为 prompt
- `tag` 为 `no_mask` 的 `text` 为目标文本

#### 3.1 Prompt 设计

这里先看一下 prompt 的设计。

PaddleOCR-VL 支持的 prompt 为：

```python
CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
```

模型原生不支持信息抽取，这里设计的 prompt 为：

```text
# 抽取全量信息
OCR:{}

# 特定值为字符串，如 `{"发票编码":"123456"}`
OCR:{key:""}

# 特定值为字典，如 `{"购买方":{"名称":"A公司"}}`
OCR:{key:{}}

# 特定值为列表，如 `{"货物或应税劳务、服务名称":[{"名称":"A产品"},{"名称":"B产品"}]}`
OCR:{key:[]}

```

可以对比一下 GLM-OCR 与 HunyuanOCR 的 prompt 的设计：

GLM-OCR 为：

```text
请按下列JSON格式输出图中信息:{key:"", ...}
```

HunyuanOCR 为：

```text
从图片中提取字段内容: [key1, key2, ...] 并以 JSON 格式返回。
```

PaddleOCR-VL-XFUND 的 prompt 的设计相对两者的区别是：

- PaddleOCR-VL-XFUND 支持全量信息抽取，也就是说，不需要指定 key
- PaddleOCR-VL-XFUND 支持列表的抽取
- PaddleOCR-VL-XFUND 的 prompt 设计的更简短，并且与 PaddleOCR-VL 的 prompt `OCR：` 有继承关系

#### 3.2 训练集的构建

XFUND 数据集共有：

- de，es，fr，it，ja，pt，zh 这七种语言的表单数据
- 每一种语言有 `149` 条训练数据，`50` 条验证数据

这样的数据量对于模型微调来说有点太少了，即便使用所有七种语言进行微调，也只有 `149*7=1043` 条数据。

因此，这里采用 `全量+子集` 的方式进行训练集的构建。

比如，`{"购买方":{"名称":"A公司"}}` 这样的一条数据，可以构建至少两条数据：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "path/to/image_0.jpg"},
    ],
    "text_info": [
        {"text": "OCR:{}", "tag": "mask"},
        {"text": "{\"购买方\":{\"名称\":\"A公司\"}}", "tag": "no_mask"},
    ]
}
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "path/to/image_0.jpg"},
    ],
    "text_info": [
        {"text": "OCR:{\"购买方\":{\"名称\":\"\"}}", "tag": "mask"},
        {"text": "{\"购买方\":{\"名称\":\"A公司\"}}", "tag": "no_mask"},
    ]
}
```

其中第一条为全量抽取，第二条为指定 KEY 进行抽取。

通过这种方式进行训练数据集的构建，得到 `6238` 条记录。（每条原始记录 1 条，加上随机再抽取子集 5 条，去处重复纪录后得到）

验证集则直接使用 XFUND 的 val 部分全量构建，共 `350` 条记录。

---

## ✨️ 模型微调

微调的过程与可以参考 [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md)

首先安装 ERNIE：

```bash
cd paddleocr_vl
git clone https://gitee.com/PaddlePaddle/ERNIE
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

然后，修改配置文件并复制覆盖原有配置文件：

```bash
cp work/sft_config/run_ocr_vl_sft_16k.yaml \
  work/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

> **注意**：由于 XFUND 的原始图片较大，这里需要注意 batch_size 的设置，或者需要适当的缩小图片

下载 PaddleOCR-VL-Receipt 模型，这里使用 aistudio：

```bash
aistudio download --model megemini/PaddleOCR-VL-Receipt --local_dir work/PaddleOCR-VL-Receipt
```

最后，就是执行微调命令即可，在 AI Studio 的 A100 环境中进行微调。

> V100 环境无法执行微调，但是可以进行模型推理

```bash
cd work/ERNIE; CUDA_VISIBLE_DEVICES=0 python -m erniekit.launcher train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

以下是训练的日志：

![](https://ai-studio-static-online.cdn.bcebos.com/5fd868397b4e4faab9c9dc66dcc8d4cc80482741d9c84d6a864cb6ee51141855)

比较于 PaddleOCR-VL-Receipt 的训练：

![logs](https://ai-studio-static-online.cdn.bcebos.com/e792644cb32f4ba79fdcb0ca87e3aba6788d3ff145e44a4e804ba9219b180e64)

- 训练的 epoch 为 200，根据训练集的数量进行设置
- loss 的初始值小于 PaddleOCR-VL-Receipt ，说明基础模型就具备一定的 XFUND 数据信息抽取能力
- loss 的最终值小于 PaddleOCR-VL-Receipt ，说明最终效果也应该好一些

---

## ✨️ 模型推理

微调完成后，可以使用微调后的模型进行推理。模型可以：

1. 输出 `JSON` 格式的完整信息
2. 根据不同的输入字段，输出对应的 `JSON` 格式的信息

具体的推理步骤在 [微调 PaddleOCR-VL 新姿势 -- Prompt 与 信息抽取](https://aistudio.baidu.com/projectdetail/9857242) 这篇文章中已经有详细的介绍，这里只单独安利一下自己开发的小工具：

## ✨️ 使用 PaddleOCR-VL-REC 进行信息抽取

可以使用 [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) 进行信息抽取：

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# 初始化识别器
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# 使用 dict 作为 query（会被转化为 JSON 字符串）
# 返回 JSON 格式（使用 json_repair 解析结果）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json 是一个字典对象
print(type(result_json))  # <class 'dict'>
print(result_json)

# 使用 list 作为 query（会被转化为 {"item1":"", "item2":""} 的形式）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["item1", "item2"],
    return_json=True
)
print(result_json)

recognizer.close()

```

---

## ✨️ 评测对比与分析

这里比较了微调后的模型 Paddle-OCR-XFUND 与 GLM-OCR 和 HunyuanOCR 的结果。

参考文章开头的图表与介绍，这里就不再赘述了。

只是单独再说明几点：

- GLM-OCR 和 HunyuanOCR 不能使用类似 `OCR:{}` 的方式进行全量信息抽取，因此，只对比了指定 KEY 的情况
- 几个模型在中文语境中的表现最好，PaddleOCR-VL-XFUND 在其他语种的优势更明显，说明其基础模型对于多语种应该有更好的支持
- 分数普遍不高的主要原因是，这里采用的是全字段的比对，整个预测的句子中只要有一个字符的错误，或者结构错误，就不能计入分数。
  并且，由于这里的数据的标注是通过大模型生成的，有可能本身也会带来一些不确定性因素。

这里列举几个比较典型的错误：

字符错误：

![](https://ai-studio-static-online.cdn.bcebos.com/5d356bbc15754c5f9db0f3955183b004ae165e1b77f64708a61bcffd9fd2416f)

列表问题，典型的语言模型幻觉问题：

![](https://ai-studio-static-online.cdn.bcebos.com/93a6d5f6ee8541d1bcba0eb2bd8088ae1e2b4805e9c346cd9cfb3a29f71db904)

符号输出不统一：

![](https://ai-studio-static-online.cdn.bcebos.com/6694c61394f94e808293d196c6e2e214f2e5811bcb334c4eb940343cdac3bf26)

结构缺失：

![](https://ai-studio-static-online.cdn.bcebos.com/842dd7c42328404da8395ce861ee1dc449d1686b31ae42ab952f77b96d774701)

结构输出匹配错误，导致评分为 0 ：

![](https://ai-studio-static-online.cdn.bcebos.com/564ac6f8aa154bf78a631fe9b66152cbbccce530dff345c588943d89f736fcf9)

Non-stop loop 无法结束，有可能是因为，评测的时候使用的图片是 50% 大小进行缩小的，导致图片模糊：

![](https://ai-studio-static-online.cdn.bcebos.com/5e7cef5b94144304b064e17d44e10e79cd567309889d456c9a46473bc6d59cb4)

这里就不再一一列举了，具体的情况可以参考日志文件。

---

## ✨️ 总结

本文介绍了如何通过微调 PaddleOCR-VL 模型完成 XFUND 数据集的信息抽取任务。

对比了 GLM-OCR 和 HunyuanOCR 的评测结果，表明 PaddleOCR-VL-XFUND 具有更好的处理能力。

未来，可以进一步分析如何更好的生成标注的数据并进行模型的训练，并提出更好的模型准确度评测方法。

---

## ✨️ 附录

### 执行命令

1. 使用 `parse_xfund.py` 整理处 KV pair

```shell
python parse_xfund.py --input dataset/xfund/fr.train.json --output dataset/xfund/fr.train.kv.json 2>&1 | head -5
```

2. 使用 `llm_kie.py` 生成完整的 jsonl 数据

```shell
python llm_kie.py --batch fr.val
```

3. 使用 `process_extracted_dataset.py` 生成 SFT 格式的数据

```shell
python work/tools/process_extracted_dataset.py /home/aistudio/work/dataset/xfund_kv -o /home/aistudio/work/dataset/xfund_extracted -r /home/aistudio/work/dataset/xfund
```

4. 使用 `merge_extracted_jsonl.py` 生成 train 与 val 数据集

```shell
python work/tools/merge_extracted_jsonl.py /home/aistudio/work/dataset/xfund_extracted -o /home/aistudio/work/dataset/train.jsonl -e 5 -s train
```

5. 使用 `evaluate_ocr.py` 进行模型评测

```shell
python evaluate_ocr.py --jsonl work/val.jsonl --model paddleocr --backend vllm \
    --model_path megemini/PaddleOCR-VL-XFUND \
    --output_dir paddleocr_vl_xfund_evaluation/ \
    --resize_ratio 0.5

python evaluate_ocr.py --jsonl work/val.jsonl --model glm --backend transformers \
    --output_dir glm_evaluation \
    --resize_ratio 0.5

pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
python evaluate_ocr.py --jsonl work/val.jsonl --model hunyuan --backend vllm \
    --output_dir hunyuan_evaluation/ \
    --resize_ratio 0.5
```


### 目录文件
```shell
$ tree -L 1 work/
work/
├── ERNIE # git clone ERNIE 到此处
├── PaddleOCR-VL-REC # PaddleOCR-VL-REC 工具
├── PaddleOCR-VL-Receipt # 基座模型
├── PaddleOCR-VL-XFUND # 微调后的模型
├── dataset # 数据集
├── evaluation # 评测结果
├── sft_config # 微调模型的配置文件
└── tools # 工具与脚本
```


```shell
$ tree work/tools/ # 工具与脚本，详见之前的执行命令
work/tools/
├── evaluate_ocr.py
├── llm_kie.py
├── merge_extracted_jsonl.py # 
├── parse_xfund.py
└── process_extracted_dataset.py
```

```shell
$ tree -L 2 work/dataset/
work/dataset/
├── train.jsonl # 训练集
├── val.jsonl # 评测集
├── xfund # xfund 数据直接放到本目录
│   ├── de.train
│   ├── de.train.json
│   ├── de.val
│   ├── de.val.json
│   ├── es.train
│   ├── es.train.json
│   ├── es.val
│   ├── es.val.json
│   ├── fr.train
│   ├── fr.train.json
│   ├── fr.val
│   ├── fr.val.json
│   ├── it.train
│   ├── it.train.json
│   ├── it.val
│   ├── it.val.json
│   ├── ja.train
│   ├── ja.train.json
│   ├── ja.val
│   ├── ja.val.json
│   ├── pt.train
│   ├── pt.train.json
│   ├── pt.val
│   ├── pt.val.json
│   ├── zh.train
│   ├── zh.train.json
│   ├── zh.val
│   └── zh.val.json
├── xfund_extracted # 用于生成训练集与评测集
│   ├── de.train.jsonl
│   ├── de.val.jsonl
│   ├── es.train.jsonl
│   ├── es.val.jsonl
│   ├── fr.train.jsonl
│   ├── fr.val.jsonl
│   ├── it.train.jsonl
│   ├── it.val.jsonl
│   ├── ja.train.jsonl
│   ├── ja.val.jsonl
│   ├── pt.train.jsonl
│   ├── pt.val.jsonl
│   ├── zh.train.jsonl
│   └── zh.val.jsonl
└── xfund_kv # KV pair 与 llm 整理的 jsonl 数据
    ├── de.train.extracted.jsonl
    ├── de.train.kv.json
    ├── de.val.extracted.jsonl
    ├── de.val.kv.json
    ├── es.train.extracted.jsonl
    ├── es.train.kv.json
    ├── es.val.extracted.jsonl
    ├── es.val.kv.json
    ├── fr.train.extracted.jsonl
    ├── fr.train.kv.json
    ├── fr.val.extracted.jsonl
    ├── fr.val.kv.json
    ├── it.train.extracted.jsonl
    ├── it.train.kv.json
    ├── it.val.extracted.jsonl
    ├── it.val.kv.json
    ├── ja.train.extracted.jsonl
    ├── ja.train.kv.json
    ├── ja.val.extracted.jsonl
    ├── ja.val.kv.json
    ├── pt.train.extracted.jsonl
    ├── pt.train.kv.json
    ├── pt.val.extracted.jsonl
    ├── pt.val.kv.json
    ├── zh.train.extracted.jsonl
    ├── zh.train.kv.json
    ├── zh.val.extracted.jsonl
    └── zh.val.kv.json
```
