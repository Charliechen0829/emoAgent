import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import json
import time
import mysql.connector
from mysql.connector import Error
import bcrypt
from datetime import datetime, timedelta
import schedule
import threading

# ===================== 数据库配置 =====================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'emoAgent_db',
    'auth_plugin': 'mysql_native_password'
}


# ===================== 系统配置 =====================
class Config:
    # 情感模型参数
    EMOTION_MODEL = "bhadresh-savani/bert-base-go-emotion"
    MAX_LENGTH = 256  # 文本最大长度

    # DeepSeek API参数
    DEEPSEEK_API_KEY = "your_personal_api_key"
    DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
    DEEPSEEK_MODEL = "deepseek-chat"

    # 系统参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMOTION_MODEL_PATH = "./saved_models/emotion_model.bin"

    # 情感标签
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]


# ===================== 数据库助手类 =====================
class DatabaseHelper:
    def __init__(self, config):
        self.config = config

    def get_connection(self):
        try:
            return mysql.connector.connect(**self.config)
        except Error as e:
            print(f"数据库连接失败: {e}")
            return None

    def authenticate_user(self, username, password):
        """用户认证，返回用户ID"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT user_id, password_hash FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                if user and bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
                    return user['user_id']
            except Error as e:
                print(f"认证失败: {e}")
            finally:
                conn.close()
        return None

    def register_user(self, username, password, age=None, gender=None, occupation=None):
        """注册新用户"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
                query = """
                INSERT INTO users (username, password_hash, age, gender, occupation)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (username, hashed_pw, age, gender, occupation))
                conn.commit()
                return cursor.lastrowid
            except Error as e:
                print(f"注册失败: {e}")
            finally:
                conn.close()
        return None

    def save_conversation(self, user_id, input_text, emotion_json, counseling_response):
        """保存对话记录"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                query = """
                INSERT INTO conversation_history 
                (user_id, input_text, emotion_json, counseling_response) 
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (
                    user_id,
                    input_text,
                    json.dumps(emotion_json),
                    counseling_response
                ))
                conn.commit()
                return True
            except Error as e:
                print(f"保存对话失败: {e}")
            finally:
                conn.close()
        return False

    def get_user_conversations(self, user_id, days=None):
        """获取用户对话记录"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                if days:
                    query = """
                    SELECT input_text, emotion_json, counseling_response, timestamp 
                    FROM conversation_history 
                    WHERE user_id = %s 
                    AND timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                    ORDER BY timestamp DESC
                    """
                    cursor.execute(query, (user_id, days))
                else:
                    query = """
                    SELECT input_text, emotion_json, counseling_response, timestamp 
                    FROM conversation_history 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC
                    """
                    cursor.execute(query, (user_id,))
                return cursor.fetchall()
            except Error as e:
                print(f"获取对话失败: {e}")
            finally:
                conn.close()
        return []

    def get_emotion_distribution(self, user_id):
        """获取用户所有历史对话的情绪分布"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT emotion_json 
                    FROM conversation_history 
                    WHERE user_id = %s
                """, (user_id,))
                sessions = cursor.fetchall()

                if not sessions:
                    return None, 0

                # 分析情绪分布
                emotion_counter = {label: 0 for label in Config.EMOTION_LABELS}
                for session in sessions:
                    emotions = json.loads(session['emotion_json'])['emotions']
                    if emotions:
                        # 取第一个（最显著）情感
                        primary_emotion = emotions[0]['label']
                        emotion_counter[primary_emotion] += 1

                return emotion_counter, len(sessions)
            except Error as e:
                print(f"获取情绪分布失败: {e}")
            finally:
                conn.close()
        return None, 0

    def save_report(self, user_id, report_title, emotion_counter, session_count, report_content):
        """保存生成的报告到数据库"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)  # 关键修复：使用字典游标

                # 获取最早和最晚对话日期
                cursor.execute("""
                    SELECT MIN(timestamp) as first_date, MAX(timestamp) as last_date
                    FROM conversation_history 
                    WHERE user_id = %s
                """, (user_id,))
                date_range = cursor.fetchone()

                # 找到主导情感
                dominant_emotion = max(emotion_counter, key=emotion_counter.get) if emotion_counter else "无数据"

                insert_query = """
                INSERT INTO weekly_reports 
                (user_id, week_start, week_end, emotion_summary, dominant_emotion, conversation_count, report_content, title)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(insert_query, (
                    user_id,
                    date_range['first_date'] if date_range else datetime.now().date(),
                    date_range['last_date'] if date_range else datetime.now().date(),
                    json.dumps(emotion_counter),
                    dominant_emotion,
                    session_count,
                    report_content,
                    report_title
                ))
                conn.commit()
                return True
            except Error as e:
                print(f"保存报告失败: {e}")
            finally:
                conn.close()
        return False


# ===================== 情感分析模块 =====================
class EmotionAnalyzer:
    def __init__(self, config):
        self.config = config
        self.tokenizer, self.model = self.load_model()

    def load_model(self):
        # 尝试加载微调模型，否则使用基础模型
        if os.path.exists(self.config.EMOTION_MODEL_PATH):
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.EMOTION_MODEL,
                num_labels=len(self.config.EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
            model.load_state_dict(torch.load(self.config.EMOTION_MODEL_PATH))
            print(f"加载微调模型: {self.config.EMOTION_MODEL_PATH}")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.EMOTION_MODEL,
                num_labels=len(self.config.EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
            print(f"加载基础模型: {self.config.EMOTION_MODEL}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.EMOTION_MODEL)
        model = model.to(self.config.DEVICE)
        return tokenizer, model

    def predict_emotion(self, text, top_k=3):
        # 预测文本情感
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.MAX_LENGTH,
            padding=True
        ).to(self.config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 获取top_k情感
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.config.EMOTION_LABELS[i], float(probs[i])) for i in top_indices]


# ===================== 心理咨询模块 =====================
class MentalHealthCounselor:
    def __init__(self, config):
        self.config = config
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
        }

    def generate_counseling(self, text, emotions):
        # 创建情感描述字符串
        # 创建情感描述字符串
        emotion_desc = ", ".join([f"{label}({prob:.1%})" for label, prob in emotions])

        # 优化后的英文提示词（移除了Answer格式要求）
        system_prompt = """\
                You are a licensed clinical psychologist with 15 years of experience.
                Generate warm, affirming psychological advice in one continuous paragraph (≤120 words).
                Focus on:
                1. Acknowledging the expressed emotions
                2. Validating the user's experience
                3. Offering practical coping strategies
                4. Ending with an empowering affirmation

                Important: 
                * Do not include any thinking process or internal reasoning
                * Provide only the final counseling advice
                * Maintain professional yet compassionate tone
                """

        # 英文示例
        fewshot_user = """\
                Text: I couldn't sleep tonight, my mind was full of images of failure, so frustrating.
                Emotions: sadness(72%), fear(55%), disappointment(48%)
                """

        fewshot_assistant = """\
                I can feel your frustration and worry as you keep recalling the failures. First, allow yourself a gentle permission—failure doesn't mean low self-worth. Try writing down the negative images, then write encouragement you'd give a friend. Practice 4-7-8 breathing to relax. Remember, failures are stepping stones to growth. You're already on the path and deserve affirmation.
                """

        # 构造当前用户请求
        user_prompt = f"""\
        text: {text}
        emotion: {emotion_desc}
        Please provide psychological counseling advice in English.
        """

        # 构造完整的请求数据
        data = {
            "model": self.config.DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": fewshot_user
                },
                {
                    "role": "assistant",
                    "content": fewshot_assistant
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False
        }

        # 发送API请求
        response = requests.post(
            self.config.DEEPSEEK_URL,
            headers=self.headers,
            json=data
        )

        if response.status_code == 200:
            # 提取最终回复部分
            full_response = response.json()['choices'][0]['message']['content']
            if "答案:" in full_response:
                return full_response.split("答案:", 1)[1].strip()
            return full_response
        else:
            raise Exception(f"DeepSeek API请求失败: {response.status_code}")

    def generate_custom_report(self, emotion_summary, report_title="emotion report", session_count=0):
        # 构造情感分布描述
        emotion_desc = "\n".join(
            [f"- {emotion}: {count}times" for emotion, count in emotion_summary.items() if count > 0])

        # 优化后的系统提示词（移除了Answer格式要求）
        system_prompt = """\
                You are a psychological counselor with 10 years of clinical experience.
                Write a 120-150 word English emotional report in one continuous paragraph.
                Structure naturally:
                1. Data review summary
                2. Gentle evaluation (acknowledge strengths and challenges)
                3. Personalized advice
                4. Empowering closing

                Important:
                * Do not include any thinking process or internal reasoning
                * Provide only the final report content
                * Use warm, professional tone
                """

        # 优化后的英文示例（移除了Answer格式）
        fewshot_user = """\
                <Data>
                Period: 2025-07-21 — 2025-07-27
                Conversation Count: 12
                Emotion Distribution:
                joy: 4 times
                sadness: 3 times
                anxiety: 2 times
                surprise: 1 time
                neutral: 2 times
                </Data>"""

        fewshot_assistant = """\
                In 12 conversations this week, you mentioned joy 4 times, showing you find happiness in busy life. Meanwhile, 3 sadness and 2 anxiety moments reveal occasional emotional dips. Your growing emotional awareness is commendable. Next week, continue your "joy journal" and practice abdominal breathing when stressed. This will boost your joy ratio and help cloudy moments pass faster.
                """

        # 当前用户数据（使用文档1的格式）
        user_data_prompt = f"""\
        <data>
        Cycle:First Dialogue — Present
        Number of conversations:{session_count}
        Emotional distribution:
        {emotion_desc}
        </data>"""

        data = {
            "model": self.config.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fewshot_user},
                {"role": "assistant", "content": fewshot_assistant},
                {"role": "user", "content": user_data_prompt},
            ],
            "stream": False,
        }

        response = requests.post(
            self.config.DEEPSEEK_URL,
            headers=self.headers,
            json=data
        )

        if response.status_code == 200:
            full_response = response.json()['choices'][0]['message']['content']
            # 提取"答案:"后的内容
            if "答案:" in full_response:
                return full_response.split("答案:", 1)[1].strip()
            return full_response
        else:
            return "报告生成中，请稍后再查看。"


# ===================== 主系统 =====================
class EmotionAnalysisAgent:
    def __init__(self):
        self.config = Config()
        self.db_helper = DatabaseHelper(DB_CONFIG)
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        self.counselor = MentalHealthCounselor(self.config)
        self.current_user_id = None
        self.guest_counter = 0
        print(f"情感分析代理已初始化 | 设备: {self.config.DEVICE}")

    def login(self, username, password):
        """用户登录"""
        self.current_user_id = self.db_helper.authenticate_user(username, password)
        if self.current_user_id:
            print(f"用户 {username} 登录成功")
            return True
        else:
            print("登录失败，用户名或密码错误")
            return False

    def register(self, username, password, age=None, gender=None, occupation=None):
        """用户注册"""
        user_id = self.db_helper.register_user(username, password, age, gender, occupation)
        if user_id:
            print(f"用户 {username} 注册成功")
            self.current_user_id = user_id
            return True
        return False

    def process_text(self, text, user_id=None):
        """处理用户输入并返回结果"""
        if user_id == "guest":
            user_id = f"guest_{self.guest_counter}"
            self.guest_counter += 1
        start_time = time.time()

        # 情感分析
        emotions = self.emotion_analyzer.predict_emotion(text)
        analysis_time = time.time() - start_time

        # 生成JSON结果
        emotion_json = {
            "text": text,
            "emotions": [
                {"label": label, "score": score}
                for label, score in emotions
            ],
            "analysis_time": f"{analysis_time:.3f}s"
        }

        # 心理咨询建议
        counseling = self.counselor.generate_counseling(text, emotions)
        total_time = time.time() - start_time

        # 保存对话记录（如果用户已登录）
        if user_id:  # 使用传入的 user_id
            self.db_helper.save_conversation(
                user_id,  # 使用参数值
                text,
                emotion_json,
                counseling
            )

            # 游客不存数据库
        if user_id and not user_id.startswith("guest"):
            self.db_helper.save_conversation(user_id, text, emotion_json, counseling)

        return {
            "analysis": emotion_json,
            "counseling": counseling,
            "total_time": f"{total_time:.3f}s"
        }

    def generate_custom_report(self, user_id):
        """生成并返回自定义报告"""
        if not user_id:
            print("请先登录")
            return "请先登录系统"

        # 获取情感分布数据
        emotion_counter, session_count = self.db_helper.get_emotion_distribution(user_id)

        if session_count == 0:
            return "无足够数据生成报告"

        report_title = "情绪周报"

        # 使用DeepSeek生成更丰富的报告内容
        enhanced_report = self.counselor.generate_custom_report(
            emotion_counter,
            report_title,
            session_count
        )

        # 保存报告到数据库
        self.db_helper.save_report(
            self.current_user_id,
            report_title,
            emotion_counter,
            session_count,
            enhanced_report
        )

        return enhanced_report

    def schedule_weekly_reports(self):
        """安排每周自动生成报告"""
        schedule.every().sunday.at("03:00").do(self._generate_reports_for_all_users)

        # 启动调度线程
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("已启动周报自动生成任务")

    def _generate_reports_for_all_users(self):
        """为所有用户生成周报"""
        conn = self.db_helper.get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT user_id FROM users")
                users = cursor.fetchall()

                for user in users:
                    self.current_user_id = user['user_id']
                    self.generate_custom_report("每周情绪报告")
                    print(f"已为用户 {user['user_id']} 生成周报")
            except Error as e:
                print(f"批量生成周报失败: {e}")
            finally:
                conn.close()


# ===================== 主程序 =====================
if __name__ == "__main__":
    agent = EmotionAnalysisAgent()

    # 启动周报自动生成任务
    agent.schedule_weekly_reports()

    # 用户认证流程
    authenticated = False
    while not authenticated:
        print("\n===== 用户认证 =====")
        action = input("请选择操作: [1]登录 [2]注册 [3]游客访问 > ").strip()

        if action == '1':  # 登录
            username = input("用户名: ").strip()
            password = input("密码: ").strip()
            authenticated = agent.login(username, password)

        elif action == '2':  # 注册
            username = input("用户名: ").strip()
            password = input("密码: ").strip()
            age = input("年龄(可选): ").strip() or None
            gender = input("性别(可选: male/female/prefer_not_to_say): ").strip() or None
            occupation = input("职业(可选): ").strip() or None

            authenticated = agent.register(username, password, age, gender, occupation)

        elif action == '3':  # 游客访问
            print("您将以游客身份使用系统，历史记录不会被保存")
            authenticated = True

        else:
            print("无效选择，请重新输入")

    print("\n情感分析代理系统已就绪！")
    print("输入英文文本进行分析，输入 'report' 生成情绪报告，输入 'exit' 退出程序")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n请输入文本: ").strip()

            if user_input.lower() == 'exit':
                print("程序已退出。")
                break

            if user_input.lower() == 'report':
                report_title = input("请输入报告标题(可选，默认为'情绪报告'): ").strip() or "情绪报告"
                report = agent.generate_custom_report(report_title)
                print(f"\n===== {report_title} =====")
                print(report)
                print("-" * 50)
                continue

            if not user_input:
                print("输入不能为空!")
                continue

            # 处理文本并输出结果
            result = agent.process_text(user_input, user_id=agent.current_user_id)

            print("\n===== 情感分析结果 =====")
            print(json.dumps(result['analysis'], indent=2, ensure_ascii=False))

            print("\n===== 心理咨询建议 =====")
            print(result['counseling'])

            print(f"\n总处理时间: {result['total_time']}")
            print("-" * 50)

        except Exception as e:
            print(f"处理时出错: {str(e)}")
