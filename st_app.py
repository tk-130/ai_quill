# st_app.py
''' **ChatAPI APIを用いた記事執筆支援モジュール**

Streamlit経由で簡易的なWebアプリとして動作する

使用方法：
    ・下記のように実行すると、ローカルでWebサーバが起動し、Webブラウザにリダイレクトされる
        >>> streamlit run  .\\st_app.py --server.port 5678

'''
import streamlit as st
from streamlit_chat import message
import asyncio
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from typing import Any, Dict, List, Optional, Union
import sys
            
class MyStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()    

class MyStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""        

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Not Print out the prompts."""
            
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        with self.tokens_area:
            st.markdown(self.tokens_stream)
            
class AiQuill:
    ''' 記事執筆支援アプリの実装クラス

    ブラウザで入力された文字列を受け取りChatGPT APIの呼び出しを行う

    Attributes
    ----------
    None
    '''

    def __init__(self) -> None:
        ''' コンストラクタ

        環境変数に設定されたOpenAIのAPIキーを設定する
        '''
        openai.api_key = os.environ['OPENAI_API_KEY']
        st.set_page_config(initial_sidebar_state='collapsed')        

    def create_prompt(self) -> str:
        ''' プロンプト作成処理

        ChatGPTの動作の前提を表すシステムプロンプト、過去のやりとり、
        ユーザ入力をAPIに渡せる形に整形し、返却する

        Returns
        ----------
        str
            APIに渡すプロンプト(文章)
        '''
        system_message = 'あなたはプロのテクニカルライターです。  \
                下記の項目に従って記事を考えてください。 \
                また、不明点についてはその都度質問してください。 \
                記載する項目: \
                ・{input}とは \
                ・{input}の種類 \
                ・{input}のメリット・デメリット \
                ・{input}のやり方(使い方) \
                ・まとめ \
               制約条件： \
                ・小学生にも分かる(ただし、その事実を記事には明示しない) \
                ・Markdownで出力すること \
                '
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name='history'),
            HumanMessagePromptTemplate.from_template('{input}')
        ])
        return prompt

    def load_conversation(self, **kwargs) -> ConversationChain:
        ''' 会話の実行処理

        指定された引数を元にChatGPT APIを呼び出し、会話のやりとりを返却する

        Parameters
        ----------
        kwargs : dict
            下記APIリファレンスの引数を持つ辞書
                https://platform.openai.com/docs/api-reference/chat/create

        Returns
        ----------
        ConversationChain
            会話のやりとりを記録したオブジェクト
        '''
        llm = ChatOpenAI(
            **kwargs,
            model_name='gpt-4',
            streaming=True,
            callback_manager=CallbackManager([
                MyStreamlitCallbackHandler(),
            ]),
            verbose=True
        )
        memory = ConversationBufferWindowMemory(return_messages=True, k=8)
        conversation = ConversationChain(
            memory=memory,
            prompt=self.create_prompt(),
            llm=llm,
            verbose=False
        )
        return conversation

    def make_sidebar(self) -> dict:
        ''' サイドバーに関する処理

        ブラウザ上でサイドバーを描画し、ChatGPT APIに関するパラメータを指定するためのUIを描画する。
        また、指定された値を辞書型で返却する。

        Returns
        ----------
        dict
            下記APIリファレンスの引数を持つ辞書
                https://platform.openai.com/docs/api-reference/chat/create
        '''
        chat_args = dict()
        st.sidebar.subheader('ChatGPT APIパラメータ')
        chat_args['temperature'] = st.sidebar.slider(key='temperature',
                                                     label='文章のランダム性:(0-2)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        chat_args['top_p'] = st.sidebar.slider(key='top_p',
                                               label='文章の多様性:(0-1)', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        chat_args['stop'] = st.sidebar.text_input(key='stop',
                                                  label='終了条件', value=None)
        chat_args['max_tokens'] = st.sidebar.number_input(key='max_tokens',
                                                          label='最大トークン数(0-)', min_value=0, value=1024)
        chat_args['presence_penalty'] = st.sidebar.slider(key='pr_penalty',
                                                          label='同じ単語が繰り返し出現することの抑制:(-2-2)', min_value=-2.0,
                                                          max_value=2.0, value=0.0, step=0.1)
        chat_args['frequency_penalty'] = st.sidebar.slider(key='freq_penalty',
                                                           label='過去の予測で出現した回数に応じた単語の出現確率の引き下げ:(-2-2)',
                                                           min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        return chat_args

    def main_proc(self) -> None:
        ''' メイン処理

        タイトル、フォーム等を描画し、ユーザ操作に関連した処理を呼び出す

        '''
        container = st.container()
        container.markdown('# AI記事生成')
        container.markdown('指定されたテーマ/用語について、下記を生成します')
        container.markdown('- 〇〇とは')
        container.markdown('- 〇〇の種類')
        container.markdown('- 〇〇のメリット・デメリット')
        container.markdown('- 〇〇のやり方(使い方)')
        container.markdown('- まとめ')

        if 'generated' not in st.session_state:
            st.session_state.generated = []
        if 'past' not in st.session_state:
            st.session_state.past = []
        if 'conversation' not in st.session_state:
            st.session_state.conversation = None
        if 'chat_args' not in st.session_state:
            st.session_state.chat_args = None

        st.session_state.chat_args = self.make_sidebar()
        form = st.form('作成する記事について', clear_on_submit=True)
        user_message = form.text_input(label='テーマ/用語', value='')
        '''
        LangChainがシステムメッセージの動的変更に対応していないため無効化
        audience_type = form.radio(
            '生成した文章の分かりやすさ',
            ('誰にでも分かる(平易)', '技術者向け(難解)'), 
            horizontal=True
        )
        '''
                        
        submitted = form.form_submit_button('生成する')
        cleared = form.form_submit_button('クリア')
        if cleared:
            st.session_state.conversation = None
            st.experimental_rerun()

        if submitted and user_message != '':  
            '''
            LangChainがシステムメッセージの動的変更に対応していないため無効化            
            if audience_type == '誰にでも分かる(平易)':
                difficulty_level = '小学生にも分かる'
            else:
                difficulty_level = '技術者ならわかる'
            '''
            
            if isinstance(st.session_state.conversation, ConversationChain):
                conversation = st.session_state.conversation
            else:
                chat_args = st.session_state.chat_args
                conversation = self.load_conversation(**chat_args)
                st.session_state.conversation = conversation

            #answer = conversation.predict(input=user_message, difficulty_level=difficulty_level)
            answer = conversation.predict(input=user_message)

            # うまく動作していないので再検討する
            if answer[-1] != '。':
                #conversation.predict(input='つづけて', difficulty_level=difficulty_level)
                conversation.predict(input=user_message + 'についてつづけて')


if __name__ == '__main__':
    AiQuill().main_proc()
