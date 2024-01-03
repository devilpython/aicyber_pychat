# aicyber_pychat
基于大模型和llama index的对话项目

## 依赖的服务项目：
  [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) 
  [milvus](https://milvus.io/)
  [redis](https://redis.io/)

## 运行方式：
  python app.py

## 接口：
### 1.数据上传接口：
`url: http://localhost:9090/upload-text.do`
#### Get请求参数：
```
collection: 数据集名称
append: 是否追加，如果为true表示追加数据，否则新建（会把之前的数据删掉）
```
#### Post请求的数据：
```
上传的文件，参数名file
```
#### 返回结果：
```json
{
    "ids": [
        "doc_id_400687434ccf458d2ad7c14501231fbe",
        "doc_id_417940cdac4c27358175b8b23bdaab87",
        "doc_id_d400edd3cb66d5a4fc68f45f28ead5ce",
        "doc_id_dc410eec2860851ecae1726459ae8158",
        "doc_id_c7f7a2dc8c679bbf3953da8ff800e5c2",
        "doc_id_a639394b3c9b96d69d58f6a730ef79a1",
        "doc_id_1df4991dce6ce28b6dd953bacf46615e",
        "doc_id_48c6004d258552244d4bff6a9d95c32b",
        "doc_id_18d78909bdaaae634c0b90f258b9ca43",
        "doc_id_37e3a062fea17d9b1506f16495e6bcf9",
        "doc_id_1682e99a1f4243ad78793d187ba2d6c2",
        "doc_id_8afee248c2e9fe18e4ad87a16215c13b",
        "doc_id_041a356e54054837a285d9869eff29cc",
        "doc_id_64a8a3c2ba3c2ee043ff69cc2aa7e716",
        "doc_id_15bc5bf16ec88aa6e40cc530f381cb36",
        "doc_id_403767dc41982dd77da8b396eaafe48b",
        "doc_id_54b66d71f5894cd8690e6a518b20ca0f",
        "doc_id_4f0dca9081948e6feaf203604c4327dc",
        "doc_id_c2c688f3e1cca5b99d73d2ad45a6e54d",
        "doc_id_29c9ebe4283697b91486a98a5023922a",
        "doc_id_e7e57b9ea11f73447a886715f4b1a746",
        "doc_id_d5f55082563aa37e0cbe144ebff8f94d",
        "doc_id_906b9fd30eb6018507597e2e0754a320",
        "doc_id_323f3cca8b3864a0d2a498717e55660b",
        "doc_id_c1e242763edaa3b5bde2945ca56e77c7",
        "doc_id_e5f99c61b8a76f5090704b54ae88b611",
        "doc_id_bb7e3817f2fe388af9ad73d55c6dc408",
        "doc_id_9ee5dfe33ade4196963ed7bd7deaf54f",
        "doc_id_144ff85f4bba79c2be7e87b3315a71be",
        "doc_id_bf4b8f828c4d37b7c7855a3d58f18317",
        "doc_id_770df8297f03643bc08c5e37833ac030",
        "doc_id_9d3d313fa9a15fb2e5ceeb70444147b6",
        "doc_id_1f3920b7b358c6c77c89acf171cbd36d",
        "doc_id_2d63768449458ccc7f7091f20bf48d5a",
        "doc_id_bcb3baeba4e64ac19fb26d2f1b56d8fb",
        "doc_id_d96df0b327cb1909387d8b7164400e69",
        "doc_id_d66acbdf929f4508013f32ea062ad140",
        "doc_id_85a4997670b5078d6c1cdacba993688a",
        "doc_id_53ebed78ed832f1575fd46e8c51ff14e",
        "doc_id_7c432008adae854641a65ed1d2c88b83",
        "doc_id_f753c4bfaf477dffd0cfa4c37ddc2bc0",
        "doc_id_76f56b8aef88ccba6f8a3cde82386455",
        "doc_id_0f586494e35700fe6b937b9c2649763f",
        "doc_id_cb2fd165716ee4d02e8e8ed4483499c7",
        "doc_id_dcbc8b2792f58d360d15535ff35d7fc3",
        "doc_id_f8408cf76c34db3e0292cd746030e5a8",
        "doc_id_608f4707b05a3e044122b9e2afba7e41",
        "doc_id_6651a5ddb7f0ea31412854bd2c762795",
        "doc_id_d689476f4314d2f28eefa331a5ce0338",
        "doc_id_980ccf1d8da21d455817b9d0807f460e",
        "doc_id_d43c60a3c7adf70f200fb12f7853b362",
        "doc_id_d6e3c4480b0ab8541c4153108756ccae",
        "doc_id_9c2d063628149a1f9d57599c24304d96",
        "doc_id_4b3cf6232056126a42463d3ea0e4a7ab",
        "doc_id_83cf198b74a58de4b4bbe5c7376c16d2",
        "doc_id_2695767e988b9f2aef6789f17e76ec4b",
        "doc_id_3a50473a6b5e570845d5d844f7d9a30e",
        "doc_id_73583e172e3fa3b2ec8045820fbf5054",
        "doc_id_7e72c3050126f452308c035fe369bb98",
        "doc_id_713992d7c2c4dbae33b5cc43316152ea",
        "doc_id_fc42f5742891e0ee3e08960eb807d348",
        "doc_id_0777cd9552026b479754e88a7728eb3c",
        "doc_id_b5d04c6cf1a2cb4ff13f4bd03200b8e2",
        "doc_id_aae8de4171bf8797f9e733fec1b54b1c",
        "doc_id_ed283b1f19181e1d60e9576ee45d016e",
        "doc_id_17cd61e29d341b664b654a829d746ce8",
        "doc_id_a117e805ea078c188a95a6e3df659064",
        "doc_id_b9261d2e43c0a15ff69ba6290a283841",
        "doc_id_7feaf063235df949b30dbb3ec5a06b2c",
        "doc_id_471180fbd001d08e8b623c1bd9542b11",
        "doc_id_c8fa01354836de0a63d1610973ffa7aa",
        "doc_id_0f2e7184adaa3bc06fc08879b304d4d0",
        "doc_id_dc7e5a24da11cc52587878510ba89acb",
        "doc_id_519d5cf47744d4b3006b7918ea51dd70",
        "doc_id_cfbc2fe716099f43f81ada5f2fd7015e",
        "doc_id_17e047f745cd6ee545161dae4605c783",
        "doc_id_f01b7c74e621fd4f182e4f9ba6b9ca0f",
        "doc_id_3ded76532479e4b4519e84221f5a5516",
        "doc_id_b85268b189d61b0ff1aeb9160540bcbf"
    ],
    "message": "ok",
    "successful": true
}
```

### 2.添加文本接口：
`url: http://localhost:9090/append-text.do`
#### Post请求的数据：
```json
{
	"collection": "test",
	"text": "大家好，我是蔡徐坤。近日网络上出现关于我的诸多话题，很抱歉因此占用了大家的时间与关注。两年前我处于单身状态，与C女士有过交往，双方之间的私事已经在2021年妥善解决，彼此没有进一步的纠葛。需要在此向大家和媒体澄清的是，我和C女士的交往均属双方自愿，既不存在“女方为未成年”的情况，也不存在所谓的“强制堕胎”，不涉及违法行为。恳请相关自媒体不传谣、不信谣。这个教训对我来说是惨痛的，这两年的时间里，我也在自责和懊悔中。再次向一直支持信任我的歌迷们道歉，向一直关注我发展与成长的媒体朋友们道歉。今后，我会严格约束自己的言行，接受大众和社会监督。也请大家尊重和保护当事人特别是C女士的个人隐私。"
}
```
#### 返回结果：
```json
{
    "ids": [
        "doc_id_d28d09fca0309b4f94aede630f2705ea"
    ],
    "message": "ok",
    "successful": true
}
```

### 3.添加QA接口：
`url: http://localhost:9090/append-qa.do`
#### Post请求的数据：
```json
{
    "collection": "test",
    "question": "你叫什么名字？",
    "answer": "我叫高飞"
}
```
#### 返回结果：
```json
{
    "id": "doc_id_1901728ed2e961e83c41a5f174e90bfa",
    "message": "ok",
    "successful": true
}
```

### 4.对话接口：
`url: http://localhost:9090/chat.do`
#### Post请求的数据：
```json
{
    "temperature": 0.5,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.1,
    "user_input": "我刚才问的什么",
    "history": [
        {
            "question": "你知道鲁迅吗？",
            "answer": "是的，我知道。鲁迅是中国近代文学的重要代表人物之一，他的作品对中国现代文学产生了深远的影响。他是一位杰出的文学家、思想家和革命家，被誉为“中国文化精神的代言人”。"
        }
    ]
}
```
#### 返回结果：
```json
{
    "message": "ok",
    "output": "您之前问我：你知道鲁迅吗？我回答了您关于鲁迅的问题，并介绍了他作为中国近代文学的代表人物及其对当代文化的影响。如果您还有其他问题或需要进一步的帮助，请随时告诉我。",
    "successful": true
}
```

### 5.查询文档接口：
`url: http://localhost:9090/search-text.do`
#### Post请求的数据：
```json
{
    "radius": 0.5,
    "temperature": 2.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.8,
    "collection": "test_col",
    "question": "蔡徐坤和谁谈恋爱"
}
```
#### 返回结果：
```json
{
    "answer": "之前有女朋友叫 C 女士",
    "message": "ok",
    "model_name": "Atom-7B-Chat",
    "prompt": "Context information is below.\n---------------------\n大家好，我是蔡徐坤。近日网络上出现关于我的诸多话题，很抱歉因此占用了大家的时间与关注。两年前我处于单身状态，与C女士有过交往，双方之间的私事已经在2021年妥善解决，彼此没有进一步的纠葛。\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: 蔡徐坤和谁谈恋爱\nAnswer: ",
    "successful": true
}
```

### 6.QA对话接口：
`url: http://localhost:9090/qa.do`
#### Post请求的数据：
```json
{
    "radius": 0.6,
    "temperature": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "collection": "test",
    "question": "我是残疾人，什么条件申请无障碍改造？"
}
```
#### 返回结果：
```json
{
    "answer": "具有北辰区户籍且持有《中华人民共和国残疾人证》的残疾人可以申请无障碍改造。具体标准是按照残疾类别、等级以及家庭困难程度而定。您可以到您户籍所在地的镇街残联或者镇街便民服务中心进行咨询并提交申请材料。",
    "message": "ok",
    "model_name": "Atom-7B-Chat",
    "prompt": "Context information is below.\n---------------------\nCommon sense questions and answers\nQuestion: 我是残疾人，什么条件申请无障碍改造？\nFactual answer:具有北辰区户籍且持有《中华人民共和国残疾人证》。根据残疾类别、等级，家庭困难程度。\n\nQuestion: 想申请无障碍改造怎么申请？\nFactual answer:您携带相关证件前往您户籍所在的镇街残联或镇街便民服务中心办理。\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: 我是残疾人，什么条件申请无障碍改造？\nAnswer: ",
    "successful": true
}
```

### 7.删除向量接口：
`url: http://localhost:9090/drop-vectors.do`
#### Post请求的数据：
```json
{
	"collection": "test_col",
	"ids": ["doc_id_400687434ccf458d2ad7c14501231fbe", "doc_id_d400edd3cb66d5a4fc68f45f28ead5ce"]
}
```
#### 返回结果：
```json
{
	"message": "ok",
	"successful": true
}
```

### 8.删除集合接口：
`url: http://localhost:9090/drop-collection.do`
#### Post请求的数据：
```json
{
	"collection": "test"
}
```
#### 返回结果：
```json
{
	"message": "ok",
	"successful": true
}
```
