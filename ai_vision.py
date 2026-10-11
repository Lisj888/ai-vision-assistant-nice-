import dashscope# 本项目基于通义千问API实现多功能视觉识别
from dashscope import MultiModalConversation
import gradio as gr
import base64
import io
from PIL import Image

# 你的 API Key（注意：别发到任何聊天框里）
dashscope.api_key = "sk-ws-H.PRMPHMH.cGqI.MEYCIQCxmQaLx_5K0Sla-3VMbx9FqlcJMgy9uG852bMz8nEmegIhAOWe34IgX89_l3kJsIbVcDoxWAdUuCaOqzV4HpqDqWff"

# 不同功能对应的提示词
PROMPTS = {
    "识别数字": "这张图片里写的是什么数字？只回答一个数字，不要解释。",
    "识别植物": "这张图片里是什么植物？请说出它的中文名称和一句简介。",
    "识别动物": "这张图片里是什么动物？请说出它的中文名称和一句简介。",
    "识别菜名": "这张图片里是什么菜？请说出它的菜名和主要食材。",
    "拍照翻译": "请把图片里的文字翻译成中文，并保持原意。",
    "自由问答": "请详细描述这张图片里的内容。"
}


def recognize(img, task):
    img = img.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 根据选择的功能，取出对应的提示词
    prompt = PROMPTS.get(task, "请描述这张图片。")

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img_base64}"},
                {"text": prompt}
            ]
        }
    ]

    response = MultiModalConversation.call(model="qwen-vl-max", messages=messages)

    try:
        result = response.output.choices[0].message.content[0]["text"]
        return result
    except Exception as e:
        return f"识别失败：{str(e)}"


# 网页界面
interface = gr.Interface(
    fn=recognize,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Dropdown(
            choices=list(PROMPTS.keys()),
            value="识别数字",
            label="选择功能"
        )
    ],
    outputs=gr.Textbox(label="识别结果", lines=5),
    title="多功能 AI 视觉助手",
    description="上传一张图片，选择你要的功能，AI 会给出对应的结果。"
)

interface.launch(server_name="0.0.0.0", server_port=7860)