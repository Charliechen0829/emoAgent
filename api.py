from flask import Flask, request, jsonify
from flask_cors import CORS
from emoAgentWithSql import EmotionAnalysisAgent  # 导入情感分析核心模块


app = Flask(__name__)
CORS(app)  # 允许跨域请求[3](@ref)

# 初始化情感分析代理（单例模式）
agent = EmotionAnalysisAgent()
agent.schedule_weekly_reports()  # 启动后台任务线程


@app.route('/api/login', methods=['POST'])
def login():
    """用户登录认证[6](@ref)"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user_id = agent.db_helper.authenticate_user(username, password)
    return jsonify({"user_id": user_id}) if user_id else jsonify({"error": "认证失败"}), 401


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')
    user_id = data.get('user_id') or "guest"  # 支持游客模式

    if not text:
        return jsonify({"error": "缺少文本参数"}), 400

    try:
        result = agent.process_text(text, user_id=user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 添加异常处理


@app.route('/api/report', methods=['POST'])
def generate_report():
    """生成情绪报告"""
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少用户ID"}), 400

    try:
        report = agent.generate_custom_report(user_id)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 本地运行配置
    app.run(host='0.0.0.0', port=5000, debug=False)
