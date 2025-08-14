const API_BASE = 'http://localhost:5000/api';  // Local API address
const REPORT_API = 'http://localhost:5000/api/report';

// User login function
async function login(username, password) {
    const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    if (data.user_id) {
        localStorage.setItem('user_id', data.user_id);  // Store user ID
        return true;
    }
    throw new Error(data.error || "Login failed");
}

// Sentiment analysis request
async function analyzeText(text) {
    const userId = localStorage.getItem('user_id');
    const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text, user_id: userId })
    });
    return response.json();
}

// Generate report request
async function generateReport(title = "Emotion Report") {
    const userId = localStorage.getItem('user_id');
    const response = await fetch(`${API_BASE}/report`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: userId, title })
    });
    return response.json();
}

// Generate weekly report
async function generateWeeklyReport() {
    // Check login status
    if (!isLoggedIn()) {
        loginModal.style.display = 'flex';
        return;
    }
    
    // Add "generating" indicator
    addReportGeneratingIndicator();
    
    try {
        // Call API to generate report
        const response = await fetch(REPORT_API, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_id: localStorage.getItem('user_id')
            })
        });
        
        if (!response.ok) {
            throw new Error(`Report generation failed: ${response.status}`);
        }
        
        const reportData = await response.json();
        
        // Remove indicator
        document.querySelector('.report-generating')?.remove();
        
        // Render report
        renderReport(reportData.report);
    } catch (error) {
        document.querySelector('.report-generating')?.remove();
        addMessage(`Report generation failed: ${error.message}`, 'ai');
    }
}

// Add report generation indicator
function addReportGeneratingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.classList.add('message', 'ai-message', 'report-generating');
    
    indicatorDiv.innerHTML = `
        <div class="message-header">
            <div class="avatar ai-avatar">
                <i class="fas fa-chart-pie"></i>
            </div>
            <strong>emoAgent Report System</strong>
        </div>
        <div class="message-content">
            <p>Generating your emotion report...</p>
            <div style="display: flex; justify-content: center; padding: 20px 0;">
                <div class="loader"></div>
            </div>
            <p>This may take a few seconds, please wait</p>
        </div>
    `;
    
    messagesContainer.appendChild(indicatorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Add loading animation style
    const style = document.createElement('style');
    style.textContent = `
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #ff6b6b;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
}

// Render report
function renderReport(reportContent) {
    // Parse report content (assume report is in JSON format)
    try {
        const report = JSON.parse(reportContent);
        
        // Use template engine to render
        const template = document.getElementById('report-template').innerHTML;
        const rendered = Mustache.render(template, report);
        
        // Add to message container
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = rendered;
        messagesContainer.appendChild(tempDiv.firstElementChild);
        
        // Render chart
        renderEmotionChart(report.emotion_distribution);
    } catch (e) {
        addMessage("Failed to parse report, please try again", 'ai');
    }
}

// Render emotion distribution chart
function renderEmotionChart(emotionData) {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    
    // Extract labels and values
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    Object.keys(emotionData).forEach(emotion => {
        if (emotionData[emotion] > 0) {
            labels.push(emotion);
            data.push(emotionData[emotion]);
            backgroundColors.push(getEmotionColor(emotion));
        }
    });
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: true,
                    text: 'Emotion Distribution'
                }
            }
        }
    });
}

// Assign colors to different emotions
function getEmotionColor(emotion) {
    const colorMap = {
        'joy': '#FFD700',      // Gold
        'sadness': '#1E90FF',  // Dodger Blue
        'anger': '#FF4500',    // Orange Red
        'fear': '#9370DB',     // Medium Purple
        'surprise': '#00FA9A', // Medium Spring Green
        'love': '#FF69B4',     // Hot Pink
        'neutral': '#A9A9A9'   // Dark Gray
    };
    
    return colorMap[emotion] || '#' + Math.floor(Math.random()*16777215).toString(16);
}

// Bind weekly report button event
document.getElementById('report-btn').addEventListener('click', generateWeeklyReport);

// Add CSS styles for report messages
const style = document.createElement('style');
style.textContent = `
    .report-message {
        max-width: 95%;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    .report-period {
        margin: 10px 0;
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .emotion-chart {
        margin: 20px 0;
        height: 300px;
    }
    
    .report-summary {
        background: rgba(255, 245, 230, 0.7);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border: 1px solid rgba(255, 214, 165, 0.5);
    }
    
    .report-insights ul {
        padding-left: 20px;
        margin: 15px 0;
    }
    
    .report-insights li {
        margin-bottom: 8px;
        line-height: 1.5;
    }
`;
document.head.appendChild(style);

// Example usage
document.getElementById('submit-btn').addEventListener('click', async () => {
    try {
        const text = document.getElementById('input-text').value;
        const analysisResult = await analyzeText(text);
        console.log("Sentiment analysis result:", analysisResult);
    } catch (error) {
        console.error("API request failed:", error);
    }
});