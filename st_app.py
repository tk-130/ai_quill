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
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain, SequentialChain, LLMChain
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

class GenerateStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self) -> None:
        st.session_state.tokens_area = st.empty()
        st.session_state.tokens_stream = ''    

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Not Print out the prompts."""
            
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if 'tokens_stream' in st.session_state:
            st.session_state.tokens_stream += token
        
            if 'tokens_area' in st.session_state:
                with st.session_state.tokens_area:
                    st.markdown(st.session_state.tokens_stream)

class EvaluateStreamlitCallbackHandler(GenerateStreamlitCallbackHandler):
    def __init__(self) -> None:
        st.session_state.eval_tokens_area = st.empty()
        st.session_state.eval_tokens_stream = ''

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if 'eval_tokens_stream' in st.session_state:
            st.session_state.eval_tokens_stream += token
        
            if 'eval_tokens_area' in st.session_state:
                with st.session_state.eval_tokens_area:
                    st.markdown(st.session_state.eval_tokens_stream)

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
                ・企業のオウンドメディア向けのWeb記事であることを前提とすること \
                ・{difficulty_level}(ただし、その事実を記事には明示しない) \
                ・Markdownで出力すること \
                '
        system_prompt = PromptTemplate(template=system_message, input_variables=['input', 'difficulty_level'])
        return system_prompt

    def create_evaluate_prompt(self) -> str:
        ''' プロンプト作成処理

        ChatGPTの動作の前提を表すシステムプロンプト、過去のやりとり、
        ユーザ入力をAPIに渡せる形に整形し、返却する

        Returns
        ----------
        str
            APIに渡すプロンプト(文章)
        '''
        system_message = ' \
                ・{input}に対する{answer}の内容が必要十分かどうか第三者視点で厳しく講評してください。(ただし、その事実を記事には明示しない) \
                記載する項目： \
                ・第三者視点AIによる講評 \
                制約条件： \
                ・第三者視点AIによる講評という見出しから開始すること(ただし、その事実を記事には明示しない) \
                ・{difficulty_level}(ただし、その事実を記事には明示しない) \
                ・企業のオウンドメディア向けのWeb記事であることを前提とする(ただし、その事実を記事には明示しない) \
                ・Markdownで出力すること(ただし、その事実を記事には明示しない) \
                '
        prompt = PromptTemplate(template=system_message, input_variables=['input', 'answer', 'difficulty_level'])
        return prompt

    def load_chain(self, **kwargs) -> SequentialChain:
        ''' 会話の実行処理

        指定された引数を元にChatGPT APIを呼び出し、会話のやりとりを返却する

        Parameters
        ----------
        kwargs : dict
            下記APIリファレンスの引数を持つ辞書
                https://platform.openai.com/docs/api-reference/chat/create

        Returns
        ----------
        SequentialChain
            LLMとのやりとりを提供するオブジェクト
        '''
        generate_llm = ChatOpenAI(
            **kwargs,
            streaming=True,
            callback_manager=CallbackManager([
                GenerateStreamlitCallbackHandler(),
            ]),
            verbose=True
        )
        
        generate_chain = LLMChain(
            llm=generate_llm, 
            prompt=self.create_prompt(), output_key='answer')

        evaluate_llm = ChatOpenAI(
            **kwargs,
            streaming=True,
            callback_manager=CallbackManager([
                EvaluateStreamlitCallbackHandler(),
            ]),
            verbose=True
        )
        
        evaluate_chain = LLMChain(
            llm=evaluate_llm, 
            prompt=self.create_evaluate_prompt(), output_key='evaluate')

        chain = SequentialChain(
            chains=[generate_chain, evaluate_chain],
            input_variables=['input', 'difficulty_level'],
            output_variables=['answer', 'evaluate'],
            verbose=False
        )
        return chain

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
        chat_args['model_name'] = st.sidebar.selectbox('モデル名', ('gpt-4', 'gpt-3.5-turbo'))
        chat_args['temperature'] = st.sidebar.slider(key='temperature',
                                                     label='文章のランダム性:(0-2)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        chat_args['top_p'] = st.sidebar.slider(key='top_p',
                                               label='文章の多様性:(0-1)', min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        chat_args['stop'] = st.sidebar.text_input(key='stop',
                                                  label='終了条件', value=None)
        if chat_args['model_name'] == 'gpt-4':
            default_token = 5000
        else:
            default_token = 2000
            
        chat_args['max_tokens'] = st.sidebar.number_input(key='max_tokens',
                                                          label='最大トークン数(0-)', min_value=0, value=default_token)
        chat_args['presence_penalty'] = st.sidebar.slider(key='pr_penalty',
                                                          label='同じ単語が繰り返し出現することの抑制:(-2-2)', min_value=-2.0,
                                                          max_value=2.0, value=0.0, step=0.1)
        chat_args['frequency_penalty'] = st.sidebar.slider(key='freq_penalty',
                                                           label='過去の予測で出現した回数に応じた単語の出現確率の引き下げ:(-2-2)',
                                                           min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        return chat_args

    def init_container_stuff(self) -> None:
        ''' セッション管理されたコンテナの初期化処理

        LangChainのコールバックマネージャから呼び出される画面更新用ハンドラ関数で使用する描画用コンテナを初期化する        

        '''        
        if 'tokens_area' in st.session_state:
            st.session_state.tokens_area = st.empty()
        if 'tokens_stream' in st.session_state:
            st.session_state.tokens_stream = ''
        if 'eval_tokens_area' in st.session_state:
            st.session_state.eval_tokens_area = st.empty()
        if 'eval_tokens_stream' in st.session_state:
            st.session_state.eval_tokens_stream = ''        


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
        if 'seq_chain' not in st.session_state:
            st.session_state.seq_chain = None
        if 'chat_args' not in st.session_state:
            st.session_state.chat_args = None

        st.session_state.chat_args = self.make_sidebar()
        form = st.form('作成する記事について', clear_on_submit=True)
        user_message = form.text_input(label='テーマ/用語', value='')

        audience_type = form.radio(
            '生成した文章の分かりやすさ',
            ('誰にでも分かる(平易)', '一般的', '技術者向け(難解)'), 
            horizontal=True
        )
                        
        submitted = form.form_submit_button('生成する')
        cleared = form.form_submit_button('クリア')
        if cleared:
            self.init_container_stuff()

            st.session_state.seq_chain = None
            st.experimental_rerun()

        if submitted and user_message != '':            
            self.init_container_stuff()

            if audience_type == '誰にでも分かる(平易)':
                difficulty_level = '小学生にも分かる'
            elif audience_type == '一般的':
                difficulty_level = 'IT知識が無い大人にも分かる'
            else:
                difficulty_level = '技術者なら分かる'

            chat_args = st.session_state.chat_args
            seq_chain = self.load_chain(**chat_args)
            st.session_state.seq_chain = seq_chain

            seq_chain({'input': user_message, 'difficulty_level':difficulty_level})


if __name__ == '__main__':
    AiQuill().main_proc()
