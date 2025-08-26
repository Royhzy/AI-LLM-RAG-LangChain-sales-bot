import time

import gradio as gr

from sales_chatbot.vectors_retrieval import save_vectors_db, init_chain


def do_user(user_message, history):  # 把用户的问题消息，放到历史记录中
    history.append((user_message, None))
    return '', history


def do_it(history):  # 定义一个回调函数，
    print(history)
    question = history[-1][0]    #拿到最后一条的第一个位置
    res = bot.invoke({'input': question})   # 把用户的提问，输入给AI机器人
    resp = res['answer']
    if not resp:
        resp = '我不知道呢，请您问问人工!'

    # 最后一条历史记录中，只有用户的提问消息，没有AI的的回答
    history[-1][1] = ''
    # 流式输出
    for char in resp:
        history[-1][1] += char  # 把最后一条聊天记录的  AI的回答 追加了一个字符
        time.sleep(0.1)
        yield history


def run_gradio():
    css = """
    #bgc {background-color: #7FFFD4}
    .feedback textarea {font-size: 24px !important}
    """

    # Blocks： 自定义各种组件联合的一个函数
    with gr.Blocks(title='我的AI聊天机器人', css=css) as instance:  # 自定义
        gr.Label('房产销售AI机器人', container=False)
        chatbot = gr.Chatbot(label='AI的回答记录', height=350,
                             placeholder='<strong>AI机器人</strong><br> 你可以问任何问题')
        msg = gr.Textbox(label='请输入问题', placeholder='输入你的问题！', elem_classes='feedback', elem_id='bgc')
        clear = gr.ClearButton(value='清除聊天记录', components=[msg, chatbot])  # 清楚的按钮

        # 光标在文本输入框中，回车。 触发submit
        # 通过设置queue=False可以禁用队列，以便立即执行。
        #  在then里面：调用do_it函数，更新聊天历史，用机器人的回复替换之前创建的None消息，并逐字显示回复内容。
        msg.submit(do_user, [msg, chatbot], [msg, chatbot], queue=False).then(do_it, chatbot, chatbot)

    # 启动服务
    instance.queue()
    instance.launch(server_name='0.0.0.0', server_port=8008)   #http://127.0.0.1:8008/


def init():
    save_vectors_db()
    global bot
    bot = init_chain()


if __name__ == '__main__':
    # 初始化房产销售的AI机器人
    init()
    run_gradio()
