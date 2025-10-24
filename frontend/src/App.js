import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';

const BACKEND_URL = "http://127.0.0.1:8000";

// --- STYLES ---
const AppStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
  @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

  :root {
    --primary-bg: #0A071E;
    --secondary-bg: #1a183c;
    --text-primary: #ffffff;
    --text-secondary: #b0b0d1;
    --accent-purple: #8a2be2;
    --accent-pink: #c837c8;
    --accent-green: #00ff9b;
    --accent-blue: #3b82f6;
    --accent-orange: #ff6b35;
    --border-color: #3a3863;
  }

  body {
    margin: 0;
    font-family: 'Roboto', sans-serif;
    background-color: var(--primary-bg);
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .app-container {
    min-height: 100vh;
    width: 100%;
    background: radial-gradient(ellipse 50% 50% at top left, rgba(42, 34, 107, 0.8), transparent),
                radial-gradient(ellipse 40% 50% at bottom right, rgba(108, 38, 186, 0.5), transparent),
                #0A071E;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
  }
  
  .main-content {
    width: 100%;
    max-width: 1200px;
    padding: 8rem 2rem 2rem 2rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-grow: 1;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 5vw;
    width: 100%;
    box-sizing: border-box;
    position: absolute;
    top: 0;
    left: 0;
    z-index: 10;
  }

  .logo {
    font-size: 2rem;
    font-weight: bold;
    color: var(--text-primary);
    cursor: pointer;
  }

  .nav {
    display: flex;
    align-items: center;
  }

  .nav a {
    color: var(--text-secondary);
    text-decoration: none;
    margin-left: 2.5rem;
    font-size: 1.1rem;
    transition: color 0.3s ease;
    display: flex;
    align-items: center;
  }
  
  .nav a:hover {
    color: var(--text-primary);
  }
  
  .nav a i {
    margin-right: 0.5rem;
  }

  .btn {
    padding: 0.8rem 2rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .btn-gradient {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink));
    color: white;
  }
  
  .btn-gradient:hover {
    box-shadow: 0 0 15px var(--accent-pink);
    transform: translateY(-2px);
  }

  .btn-secondary {
    background-color: var(--accent-blue);
    color: white;
  }

  .btn-secondary:hover {
    box-shadow: 0 0 15px var(--accent-blue);
    transform: translateY(-2px);
  }
  
  .btn-green {
      background-color: var(--accent-green);
      color: #0c0a24;
  }

  .btn-green:hover {
      box-shadow: 0 0 15px var(--accent-green);
      transform: translateY(-2px);
  }

  .btn-outline {
    background: transparent;
    border: 2px solid var(--border-color);
    color: var(--text-primary);
  }

  .btn-outline:hover {
    border-color: var(--accent-purple);
    box-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
  }

  .back-link {
    display: inline-flex;
    align-items: center;
    color: var(--text-secondary);
    text-decoration: none;
    position: fixed;
    top: 6rem;
    left: 5vw;
    z-index: 10;
    font-size: 1rem;
  }

  .back-link:hover {
    color: var(--text-primary);
  }

  .back-link i {
    margin-right: 0.5rem;
  }

  .auth-form-container {
    background-color: rgba(12, 10, 36, 0.5);
    padding: 3rem;
    border-radius: 16px;
    width: 100%;
    max-width: 450px;
    text-align: center;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
    margin: auto;
    position: relative;
  }
  
  .page-wrapper {
      width: 100%;
      position: relative;
  }

  .auth-form-container h1 {
    font-size: 1.7rem;
    margin-bottom: 0.5rem;
  }

  .auth-form-container p {
    color: var(--text-secondary);
    margin-bottom: 2rem;
  }

  .social-btn {
    width: 100%;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.8rem;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    border: 1px solid var(--border-color);
  }
  
  .social-btn i {
      margin-right: 1rem;
      font-size: 1.2rem;
  }

  .google-btn {
    background-color: white;
    color: #333;
  }

  .github-btn {
    background-color: #24292e;
    color: white;
  }

  .divider {
    display: flex;
    align-items: center;
    text-align: center;
    color: var(--text-secondary);
    margin: 2rem 0;
  }

  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid var(--border-color);
  }

  .divider:not(:empty)::before {
    margin-right: .25em;
  }

  .divider:not(:empty)::after {
    margin-left: .25em;
  }

  .input-group {
    margin-bottom: 1.5rem;
    text-align: left;
  }

  .input-group label {
    display: block;
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
  }

  .input-field {
    width: 100%;
    padding: 1rem;
    background-color: var(--primary-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 1rem;
    box-sizing: border-box;
  }

  .form-link {
    color: var(--accent-green);
    text-decoration: none;
  }
  
  .form-link:hover {
      text-decoration: underline;
  }

  .text-center {
    text-align: center;
  }

  .gradient-text {
    background: linear-gradient(90deg, #3416DD, #D418D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
  }
  
  .mt-1 { margin-top: 1rem; }
  .mt-2 { margin-top: 2rem; }
  
  .upload-container {
    text-align: center;
    width: 100%;
  }
  
  .upload-box {
    background-color: rgba(12, 10, 36, 0.5);
    border: 2px dashed var(--border-color);
    border-radius: 16px;
    padding: 4rem 2rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
    max-width: 600px;
    margin: 2rem auto;
  }

  .upload-box:hover {
    background-color: rgba(26, 24, 60, 0.7);
  }
  
  .upload-box i {
      font-size: 3rem;
      color: var(--accent-purple);
      margin-bottom: 1rem;
  }
  
  .info-box {
      background-color: rgba(59, 130, 246, 0.1);
      border-left: 4px solid var(--accent-blue);
      padding: 1rem;
      border-radius: 8px;
      margin: 1rem auto;
      max-width: 600px;
      text-align: left;
  }
  
  .info-box i {
      margin-right: 0.5rem;
  }

  .file-display {
      background-color: var(--secondary-bg);
      padding: 1.5rem;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      max-width: 500px;
      margin: 2rem auto;
  }
  
  .file-display i {
      color: var(--accent-green);
      font-size: 1.5rem;
      margin-right: 1rem;
  }
  
  .dataset-info-container {
      background-color: var(--secondary-bg);
      padding: 2rem;
      border-radius: 16px;
      margin: 2rem 0;
      display: flex;
      justify-content: space-around;
  }

  .info-item {
      text-align: center;
  }

  .info-item .value {
      font-size: 2.5rem;
      font-weight: bold;
      color: var(--accent-pink);
  }

  .info-item .label {
      color: var(--text-secondary);
      margin-top: 0.5rem;
  }
  
  .explore-options {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1.5rem;
      margin: 2rem 0;
  }

  .explore-card {
      background-color: var(--secondary-bg);
      padding: 2rem;
      border-radius: 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      border: 1px solid var(--border-color);
  }
  
  .explore-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 5px 20px rgba(138, 43, 226, 0.3);
  }
  
  .explore-card i {
      font-size: 3rem;
      margin-bottom: 1rem;
      color: var(--text-primary);
  }

  .explore-card p {
      color: var(--text-primary);
      margin: 0;
      font-weight: 500;
  }
  
  .data-explorer-tabs, .dashboard-tabs {
      display: flex;
      gap: 1rem;
      margin-bottom: 2rem;
      background-color: var(--secondary-bg);
      padding: 0.5rem;
      border-radius: 12px;
      max-width: max-content;
  }

  .tab {
      padding: 0.8rem 1.5rem;
      border-radius: 8px;
      cursor: pointer;
      color: var(--text-secondary);
      font-weight: 500;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border: none;
      background: transparent;
  }

  .tab.active {
      background-color: var(--primary-bg);
      color: var(--text-primary);
      box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }
  
  .table-container {
      overflow-x: auto;
      background-color: var(--secondary-bg);
      border-radius: 12px;
      padding: 1rem;
  }
  
  table {
      width: 100%;
      border-collapse: collapse;
  }
  
  th, td {
      padding: 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
  }
  
  th {
      color: var(--text-secondary);
      text-transform: capitalize;
  }
  
  .structure-card {
      background-color: var(--secondary-bg);
      padding: 1.5rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
  }
  
  .dist-cards {
      display: flex;
      gap: 1rem;
      justify-content: center;
      margin-top: 2rem;
  }
  
  .dist-card {
      padding: 1.5rem;
      border-radius: 12px;
      text-align: center;
      flex-grow: 1;
  }

  .dist-card.numeric { background-color: #3b82f6; }
  .dist-card.text { background-color: #16a34a; }
  .dist-card.date { background-color: #c026d3; }
  
  .missing-data-card {
      background-color: var(--secondary-bg);
      padding: 3rem;
      border-radius: 16px;
      text-align: center;
  }
  
  .missing-data-card i {
      font-size: 4rem;
      color: var(--accent-green);
      margin-bottom: 1rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .dashboard-card {
    background-color: var(--secondary-bg);

    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }

  .dashboard-card h3 {
    margin-top: 0;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 1rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
  }
  
  .metric-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }
  .metric-item .value {
    color: var(--accent-green);
    font-weight: bold;
  }

  .about-page {
    text-align: center;
    max-width: 900px;
    margin: auto;
    width: 100%;
  }
  .about-page .brain-icon {
    font-size: 4rem;
    color: var(--accent-purple);
    margin-bottom: 1rem;
  }
  .about-cards {
    display: flex;
    gap: 2rem;
    margin: 3rem 0;
    text-align: left;
  }
  .about-card {
    background-color: var(--secondary-bg);
    padding: 2rem;
    border-radius: 16px;
    flex: 1;
  }
  .about-card h3 {
    margin-top: 0;
    color: var(--accent-pink);
  }

  @keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
  }

  .bot-fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 60px;
    height: 60px;
    background: linear-gradient(45deg, var(--accent-purple), var(--accent-pink));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.3s ease;
    z-index: 1000;
    animation: blink 2s infinite ease-in-out;
  }
  .bot-fab:hover {
    transform: scale(1.1);
    animation-play-state: paused;
  }
  .bot-fab i {
    font-size: 1.8rem;
    color: white;
  }

  .bot-container {
    width: 100%;
    max-width: 400px;
    height: 70vh;
    background-color: var(--secondary-bg);
    border-radius: 16px;
    margin: auto;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
  }
  .bot-header {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink));
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .bot-header img {
    width: 40px;
    height: 40px;
    border-radius: 50%;
  }
  .bot-header h3 {
    margin: 0;
    color: var(--text-primary);
  }
  .bot-messages {
    flex-grow: 1;
    padding: 1rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .message {
    padding: 0.8rem;
    border-radius: 12px;
    max-width: 80%;
    word-wrap: break-word;
  }
  .user-message {
    background-color: var(--accent-purple);
    align-self: flex-end;
  }
  .bot-message {
    background-color: var(--secondary-bg);
    align-self: flex-start;
  }
  .bot-input-form {
    display: flex;
    padding: 1rem;
    border-top: 1px solid var(--border-color);
  }
  .bot-input-form input {
    flex-grow: 1;
    background-color: var(--primary-bg);
    border: 1px solid var(--border-color);
    color: white;
    padding: 0.8rem;
    border-radius: 20px 0 0 20px;
    outline: none;
  }
  .bot-input-form button {
    width: 50px;
    border: none;
    background-color: var(--accent-purple);
    color: white;
    border-radius: 0 20px 20px 0;
    cursor: pointer;
  }

  /* New styles for AI analysis */
  .ai-analysis-card {
    background: linear-gradient(135deg, rgba(138, 43, 226, 0.1), rgba(200, 55, 200, 0.1));
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .ai-recommendation {
    background-color: rgba(26, 24, 60, 0.7);
    border-left: 4px solid var(--accent-green);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
  }

  .ai-recommendation h4 {
    margin-top: 0;
    color: var(--accent-green);
  }

  .ai-recommendation p {
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
  }

  .ai-recommendation small {
    color: var(--accent-blue);
    font-size: 0.8rem;
  }

  .chart-container {
    background: var(--secondary-bg);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    position: relative;
  }

  .chart-title {
    font-size: 1.2rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
    text-align: center;
  }

  .chart-loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px;
    color: var(--text-secondary);
  }

  .chart-error {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px;
    color: #ef4444;
    text-align: center;
    padding: 1rem;
  }

  .chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
  }

  .full-width-chart {
    grid-column: 1 / -1;
  }

  /* Plotly chart styling */
  .plotly-chart {
    width: 100%;
    height: 400px;
    background-color: var(--secondary-bg);
    border-radius: 8px;
  }

  .export-buttons {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
  }

  /* Streamlit-like styles */
  .streamlit-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
  }
  
  .streamlit-metric {
    background: var(--secondary-bg);
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
  }
  
  .streamlit-metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--accent-purple);
    margin-bottom: 0.5rem;
  }
  
  .streamlit-metric-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
  }
  
  .analysis-recommendations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }
  
  .analysis-recommendation-card {
    background: var(--secondary-bg);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid var(--accent-green);
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  .analysis-recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  }
  
  .streamlit-chat-container {
    background: var(--secondary-bg);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }
  
  .streamlit-chat-messages {
    height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
    padding: 1rem;
    background: var(--primary-bg);
    border-radius: 8px;
  }
  
  .streamlit-message {
    margin: 0.5rem 0;
    padding: 0.8rem;
    border-radius: 8px;
    max-width: 80%;
  }
  
  .streamlit-user-message {
    background: var(--accent-purple);
    color: white;
    margin-left: auto;
    text-align: right;
  }
  
  .streamlit-bot-message {
    background: var(--secondary-bg);
    color: var(--text-primary);
    margin-right: auto;
  }

  .analysis-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
  }

  .analysis-card {
    background: var(--secondary-bg);
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .analysis-card:hover {
    border-color: var(--accent-purple);
    transform: translateY(-2px);
  }

  .analysis-card.active {
    border-color: var(--accent-green);
    background: rgba(0, 255, 155, 0.1);
  }

  .analysis-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 0.5rem;
  }

  .analysis-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    font-size: 0.9rem;
  }

  .status-running {
    color: var(--accent-orange);
  }

  .status-ready {
    color: var(--accent-green);
  }

  .dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
  gap: 2rem;
  width: 100%;
}

.dashboard-card {
  background-color: var(--secondary-bg);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  width: 100%;
  box-sizing: border-box;
}

.dashboard-card h3 {
  margin-top: 0;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
  color: var(--text-primary);
  font-size: 1.3rem;
}

/* Full width chart containers */
.full-width-chart {
  grid-column: 1 / -1;
  width: 100%;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
  gap: 2rem;
  width: 100%;
}

.chart-container {
  background: var(--secondary-bg);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  width: 100%;
  box-sizing: border-box;
}

.plotly-chart {
  width: 100% !important;
  height: 500px !important;
  background-color: var(--secondary-bg);
  border-radius: 8px;
}

/* Responsive adjustments */
@media (max-width: 1200px) {
  .dashboard-grid {
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  }
  
  .chart-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .dashboard-card {
    padding: 1.5rem;
  }
  
  .plotly-chart {
    height: 400px !important;
  }
}

/* Streamlit metrics with better spacing */
.streamlit-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.5rem;
  margin: 2rem 0;
  width: 100%;
}

/* Analysis recommendations grid */
.analysis-recommendations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
  width: 100%;
}

/* Main content area expansion */
.main-content {
  width: 100%;
  max-width: 1400px; /* Increased from 1200px */
  padding: 8rem 2rem 2rem 2rem;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex-grow: 1;
}
`;

// --- DATA ANALYSIS HELPERS ---

// Simple CSV to JSON parser
const parseCSV = (text) => {
    const lines = text.replace(/\r/g, '').split('\n');
    const header = lines[0].split(',');
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i]) continue;
        const obj = {};
        const currentline = lines[i].split(',');
        for (let j = 0; j < header.length; j++) {
            obj[header[j]] = currentline[j];
        }
        data.push(obj);
    }
    return { header, data };
};

// Infer data type of a string value
const inferType = (value) => {
    if (value === null || value === undefined || value.trim() === '') return 'Empty';
    if (!isNaN(value) && value.trim() !== '') return 'Numeric';
    // A simple date check, can be improved
    if (!isNaN(Date.parse(value))) return 'Date';
    return 'Text';
};

// Calculate statistics for a column
const calculateStats = (data, columnName, type) => {
  if (!data || data.length === 0) return { stats: 'No data', details: '' };
  
  const values = data.map(row => row[columnName]).filter(val => val !== null && val !== undefined && val !== '');
  
  if (values.length === 0) return { stats: 'All values missing', details: '' };
  
  if (type === 'Numeric') {
    const numericValues = values.map(val => parseFloat(val)).filter(val => !isNaN(val));
    if (numericValues.length === 0) return { stats: 'No numeric values', details: '' };
    
    const min = Math.min(...numericValues);
    const max = Math.max(...numericValues);
    const sum = numericValues.reduce((a, b) => a + b, 0);
    const mean = sum / numericValues.length;
    const sorted = [...numericValues].sort((a, b) => a - b);
    const median = sorted.length % 2 === 0 
      ? (sorted[sorted.length/2 - 1] + sorted[sorted.length/2]) / 2 
      : sorted[Math.floor(sorted.length/2)];
    
    return {
      stats: `Min: ${min.toFixed(2)}, Max: ${max.toFixed(2)}`,
      details: `Mean: ${mean.toFixed(2)}, Median: ${median.toFixed(2)}`
    };
  } else {
    // For text columns, show value counts
    const valueCounts = {};
    values.forEach(val => {
      valueCounts[val] = (valueCounts[val] || 0) + 1;
    });
    
    const uniqueValues = Object.keys(valueCounts).length;
    const mostCommon = Object.entries(valueCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([val, count]) => `${val} (${count})`)
      .join(', ');
    
    return {
      stats: `${uniqueValues} unique values`,
      details: `Most common: ${mostCommon || 'N/A'}`
    };
  }
};

// Plotly configuration
const plotlyConfig = {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
  responsive: true
};

const plotlyLayout = {
  paper_bgcolor: '#1a183c',
  plot_bgcolor: '#1a183c',
  font: {
    color: '#ffffff'
  },
  xaxis: {
    gridcolor: '#3a3863',
    linecolor: '#3a3863',
    zerolinecolor: '#3a3863'
  },
  yaxis: {
    gridcolor: '#3a3863',
    linecolor: '#3a3863',
    zerolinecolor: '#3a3863'
  },
  margin: {
    l: 60,
    r: 40,
    b: 60,
    t: 80,
    pad: 4
  }
};

// Chart color palette
const chartColors = [
  'rgba(138, 43, 226, 0.8)',  // Purple
  'rgba(200, 55, 200, 0.8)',  // Pink
  'rgba(59, 130, 246, 0.8)',  // Blue
  'rgba(0, 255, 155, 0.8)',   // Green
  'rgba(255, 99, 132, 0.8)',  // Red
  'rgba(255, 159, 64, 0.8)',  // Orange
  'rgba(255, 205, 86, 0.8)',  // Yellow
  'rgba(75, 192, 192, 0.8)',  // Teal
  'rgba(153, 102, 255, 0.8)', // Purple
  'rgba(201, 203, 207, 0.8)'  // Gray
];

// --- COMPONENTS ---

const Header = ({ setPage, isLoggedIn, setLoggedIn }) => {
    const handleLogout = () => {
        setLoggedIn(false);
        setFile(null);
        setDataset(null);
        setPage('home');
    }
    return (
        <header className="header">
            <div className="logo gradient-text" onClick={() => setPage('home')}>AutoDash AI</div>
            <nav className="nav">
                <a href="#home" onClick={(e) => { e.preventDefault(); setPage('home'); }}><i className="fas fa-home"></i>Home</a>
                <a href="#about" onClick={(e) => { e.preventDefault(); setPage('about');}}><i className="fas fa-info-circle"></i>About Us</a>
                {isLoggedIn ? (
                     <a href="#logout" onClick={(e) => { e.preventDefault(); handleLogout();}}><i className="fas fa-sign-out-alt"></i>Logout</a>
                ) : (
                    <a href="#login" onClick={(e) => { e.preventDefault(); setPage('login');}}><i className="fas fa-sign-in-alt"></i>Login</a>
                )}
            </nav>
        </header>
    );
};

const BackButton = ({ onClick, text = "Back" }) => (
    <a href="#back" onClick={(e) => {e.preventDefault(); onClick();}} className="back-link">
        <i className="fas fa-arrow-left"></i> {text}
    </a>
);

const HomePage = ({ setPage }) => {
    const handleStart = () => {
        setPage('upload');
    };
    return (
        <div className="text-center">
            <h1 style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>
                <span style={{ color: 'var(--text-primary)' }}>Your Personal </span>
                <span className="gradient-text">Data Analyst</span>
            </h1>
            <p style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto 2.5rem auto' }}>
                AutoDash AI empowers you to effortlessly derive deep, actionable insights from your datasets. Upload, analyze, and visualize your data with intelligent automation.
            </p>
            <button onClick={handleStart} className="btn btn-gradient">Start Analysis</button>
        </div>
    );
};

const LoginPage = ({ setPage, setLoggedIn }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        if (email === 'test@example.com' && password === 'password') {
            setLoggedIn(true);
            setPage('upload');
            setError('');
        } else {
            setError('Invalid email or password');
        }
    };
    const handleGoogleLogin = () => {
        setLoggedIn(true);
        setPage('upload');
    }
    return (
        <div className="page-wrapper">
            <BackButton onClick={() => setPage('home')} />
            <div className="auth-form-container">
                <h1>
                    <span style={{ color: 'var(--text-primary)' }}>Welcome to </span> 
                    <span style={{color: 'var(--accent-purple)'}}>Auto</span>
                    <span style={{color: 'var(--accent-pink)'}}>Dash</span>
                    <span style={{ color: 'var(--text-primary)' }}> AI</span>
                </h1>
                <p>Sign in to start analyzing your data</p>
                
                {error && <p style={{color: 'red'}}>{error}</p>}
                <button className="social-btn google-btn" onClick={handleGoogleLogin}><i className="fab fa-google"></i>Continue with Google</button>
                <button className="social-btn github-btn"><i className="fab fa-github"></i>Continue with GitHub</button>
                <div className="divider">or continue with email</div>
                <form onSubmit={handleLogin}>
                    <div className="input-group">
                        <label htmlFor="email">Email</label>
                        <input type="email" id="email" className="input-field" placeholder="Enter email" required value={email} onChange={e => setEmail(e.target.value)} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" className="input-field" placeholder="Enter password" required value={password} onChange={e => setPassword(e.target.value)} />
                    </div>
                     <p style={{fontSize: '0.8rem', color: 'var(--accent-green)', textAlign: 'left', marginTop: '-1rem'}}>Hint: Use test@example.com and 'password'</p>
                    <a href="#forgot" onClick={(e) => e.preventDefault()} className="form-link" style={{float: 'right', marginBottom: '1.5rem'}}>Forgot password?</a>
                    <button type="submit" className="btn btn-gradient" style={{ width: '100%' }}>Sign in</button>
                </form>
                <p className="mt-2">Don't have an account? <a href="#signup" onClick={(e) => {e.preventDefault(); setPage('signup');}} className="form-link">Sign up</a></p>
            </div>
        </div>
    );
};

const SignUpPage = ({ setPage, setLoggedIn }) => {
    const handleSignUp = (e) => {
        e.preventDefault();
        setLoggedIn(true);
        setPage('upload');
    };
    return (
        <div className="page-wrapper">
            <BackButton onClick={() => setPage('home')} />
            <div className="auth-form-container">
                <h1>Create Your Account</h1>
                <p>Join AutoDash AI and start analyzing your data</p>
                <button className="social-btn google-btn"><i className="fab fa-google"></i>Continue with Google</button>
                <button className="social-btn github-btn"><i className="fab fa-github"></i>Continue with GitHub</button>
                <div className="divider">Or continue with email</div>
                <form onSubmit={handleSignUp}>
                    <div className="input-group">
                        <label htmlFor="fullname">Full Name</label>
                        <input type="text" id="fullname" className="input-field" placeholder="Enter your Full Name" required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="email">Email</label>
                        <input type="email" id="email" className="input-field" placeholder="Enter your email" required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" className="input-field" placeholder="Create your password" required />
                    </div>
                    <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>I agree to the <a href="#" className="form-link">Terms of Service</a> and <a href="#" className="form-link">Privacy Policy</a>.</p>
                    <button type="submit" className="btn btn-secondary" style={{ width: '100%' }}>Create Account</button>
                </form>
                <p className="mt-2">Already have an account? <a href="#login" onClick={(e) => {e.preventDefault(); setPage('login');}} className="form-link">Sign in here</a></p>
            </div>
        </div>
    );
};

const UploadDatasetPage = ({ setPage, isLoggedIn, setFile }) => {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileSelect = (selectedFile) => {
        if (selectedFile) {
            setFile(selectedFile);
        }
    };
    
    const handleButtonClick = () => {
        fileInputRef.current.click();
    };

    const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => { 
        e.preventDefault(); 
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };
    
    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    };

    return (
        <div className="page-wrapper">
            <input 
                type="file" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                onChange={handleFileChange} 
                accept=".csv,.xlsx,.xls"
            />
            <BackButton onClick={() => setPage('home')} />
            <div className="upload-container">
                <h1 style={{ color: 'var(--text-primary)', fontSize: '2.5rem' }}>Upload Your Dataset</h1>
                <p style={{ color: 'var(--text-secondary)' }}>
                    {isLoggedIn ? 
                    'Ready to analyze! Upload your CSV or Excel file to get started with intelligent insights.' : 
                    'Get started with intelligent data analysis by uploading your CSV or Excel file'}
                </p>

                {!isLoggedIn && (
                    <div className="info-box" style={{ color: 'var(--text-primary)' }}>
                       <i className="fas fa-info-circle"></i> Please log in to upload and analyze your datasets. Your data will be securely processed and stored.
                    </div>
                )}
                
                <div className="text-center" style={{marginTop: '2rem'}}>
                    <h2>Ready to Analyze Your Data?</h2>
                    <p style={{ color: 'var(--text-secondary)' }}>Upload CSV or Excel files up to 100MB. We support automatic data cleaning and type detection</p>
                </div>

                <div 
                    className="upload-box"
                    style={{ opacity: isLoggedIn ? 1 : 0.5, borderColor: isDragging ? 'var(--accent-green)' : 'var(--border-color)'}}
                    onDragOver={isLoggedIn ? handleDragOver : undefined}
                    onDragLeave={isLoggedIn ? handleDragLeave : undefined}
                    onDrop={isLoggedIn ? handleDrop : undefined}
                    onClick={isLoggedIn ? handleButtonClick : undefined}
                >
                    <i className="fas fa-upload"></i>
                    <p>Drag and drop your file here</p>
                    <p style={{ color: 'var(--text-secondary)' }}>or click to browse</p>
                    <button className="btn btn-secondary" style={{pointerEvents: 'none'}}>Choose File</button>
                </div>

                {!isLoggedIn && (
                    <div className="text-center mt-2">
                        <p style={{color: 'var(--text-secondary)'}}>You need to be logged in to upload files</p>
                        <button className="btn btn-secondary" onClick={() => setPage('login')}>
                            <i className="fas fa-sign-in-alt" style={{marginRight: '0.5rem'}}></i>Login to Continue
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

const FileUploadedView = ({ setPage, file, setDataset, setFile }) => {
    const fileInputRef = useRef(null);
    const [isLoading, setIsLoading] = useState(false);
    
    const handleAnalyze = async () => {
        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                setDataset(data);
                setPage('datasetLoaded');
            } else {
                alert('Error uploading file: ' + data.detail);
            }
        } catch (error) {
            alert('Error uploading file: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleChooseFileClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (event) => {
        const newFile = event.target.files[0];
        if (newFile) {
            setFile(newFile);
        }
    };

    return (
        <div className="page-wrapper">
            <input 
                type="file" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                onChange={handleFileChange} 
                accept=".csv,.xlsx,.xls"
            />
            <BackButton onClick={() => setFile(null)} />
            <div className="upload-container">
                <h1 style={{ color: 'var(--text-primary)', fontSize: '2.5rem' }}>Upload Your Dataset</h1>
                <p style={{ color: 'var(--text-secondary)' }}>Ready to analyze! Upload your CSV or Excel file to get started with intelligent insights.</p>
                
                <div style={{background: 'var(--secondary-bg)', padding: '3rem', borderRadius: '16px', maxWidth: '700px', margin: 'auto'}}>
                     <h2 style={{color: 'var(--text-primary)'}}>Upload Your Data File</h2>
                     <p style={{ color: 'var(--text-secondary)' }}>Drag and drop or click to select your CSV or Excel file (up to 100MB)</p>
                     
                     <div className="file-display" style={{flexDirection: 'column', gap: '1rem'}}>
                        <div>
                            <i className="fas fa-check-circle"></i>
                            <span style={{ color: 'var(--text-primary)' }}>{file.name}</span>
                        </div>
                        <button className="btn" onClick={handleChooseFileClick} style={{backgroundColor: '#eee', color: '#333', padding: '0.5rem 1rem'}}>Choose different file</button>
                     </div>

                     <button 
                        className="btn btn-green mt-2" 
                        onClick={handleAnalyze}
                        disabled={isLoading}
                     >
                        {isLoading ? (
                            <>
                                <i className="fas fa-spinner fa-spin" style={{marginRight: '0.5rem'}}></i>
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <i className="fas fa-cogs" style={{marginRight: '0.5rem'}}></i>
                                Analyze Data
                            </>
                        )}
                     </button>
                </div>
                
                <div style={{display: 'flex', gap: '2rem', maxWidth: '700px', margin: '3rem auto', textAlign: 'left'}}>
                    <div>
                        <h3>What We Support</h3>
                        <ul style={{color: 'var(--text-secondary)', paddingLeft: '1.2rem'}}>
                            <li>CSV files (.csv)</li>
                            <li>Excel files (.xlsx, .xls)</li>
                            <li>Files up to 100MB</li>
                            <li>UTF-8 and other encodings</li>
                        </ul>
                    </div>
                     <div>
                        <h3>AI-Powered Analysis</h3>
                        <ul style={{color: 'var(--text-secondary)', paddingLeft: '1.2rem'}}>
                            <li>Automatic data cleaning</li>
                            <li>Smart type detection</li>
                            <li>Missing value analysis</li>
                            <li>Instant visualizations</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

const DatasetLoadedPage = ({ setPage, dataset }) => {
    return (
        <div className="page-wrapper">
            <BackButton onClick={() => setPage('upload')} />
            <div className="text-center" style={{maxWidth: '800px', margin: 'auto'}}>
                <i className="fas fa-check-circle" style={{fontSize: '3rem', color: 'var(--accent-green)', marginBottom: '1rem'}}></i>
                <h1 style={{color: 'var(--accent-pink)'}}>Dataset Loaded Successfully!</h1>
                <p style={{color: 'var(--text-secondary)'}}>Your data has been processed and is ready for intelligent analysis. Explore insights with our comprehensive data explorer.</p>
                
                <div className="dataset-info-container">
                    <div className="info-item">
                        <div className="value">{dataset.rows}</div>
                        <div className="label">Total rows</div>
                    </div>
                    <div className="info-item">
                        <div className="value">{dataset.columns}</div>
                        <div className="label">Columns</div>
                    </div>
                    <div className="info-item">
                        <div className="value">{dataset.dataset_type}</div>
                        <div className="label">Dataset Type</div>
                    </div>
                </div>

                <h2 style={{marginTop: '3rem'}}>Explore Your Data</h2>
                <div className="explore-options">
                    <div className="explore-card" onClick={() => setPage('dataExplorer/preview')}>
                        <i className="fas fa-eye"></i>
                        <p>Preview</p>
                    </div>
                    <div className="explore-card" onClick={() => setPage('dataExplorer/structure')}>
                        <i className="fas fa-sitemap"></i>
                        <p>Structure</p>
                    </div>
                    <div className="explore-card" onClick={() => setPage('dataExplorer/missing')}>
                        <i className="fas fa-question-circle"></i>
                        <p>Missing Data</p>
                    </div>
                    <div className="explore-card" onClick={() => setPage('automatedDashboard/overview')}>
                        <i className="fas fa-chart-line"></i>
                        <p>Dashboard</p>
                    </div>
                </div>

                <button className="btn btn-secondary mt-2" onClick={() => setPage('dataExplorer/preview')}>Open Data Explorer</button>
            </div>
        </div>
    );
};

const DataExplorerPage = ({ setPage, dataset, initialTab }) => {
    const [activeTab, setActiveTab] = useState(initialTab);

    const renderContent = () => {
        switch(activeTab) {
            case 'preview': return <DataPreview dataset={dataset} />;
            case 'structure': return <DataStructure dataset={dataset} />;
            case 'missing': return <MissingData dataset={dataset} />;
            default: return <DataPreview dataset={dataset} />;
        }
    };
    
    return (
        <div className="page-wrapper" style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
            <BackButton onClick={() => setPage('datasetLoaded')} text="Back"/>
            <div style={{width: '100%'}}>
                <h1>Unlock Your Data's Potential</h1>
                <p style={{color: 'var(--text-secondary)'}}>Comprehensive analysis and visualization of your dataset with AI-powered insights</p>
            </div>

            <div className="data-explorer-tabs">
                <button className={`tab ${activeTab === 'preview' ? 'active' : ''}`} onClick={() => setActiveTab('preview')}>
                    <i className="fas fa-eye"></i> Preview
                </button>
                <button className={`tab ${activeTab === 'structure' ? 'active' : ''}`} onClick={() => setActiveTab('structure')}>
                    <i className="fas fa-sitemap"></i> Structure
                </button>
                <button className={`tab ${activeTab === 'missing' ? 'active' : ''}`} onClick={() => setActiveTab('missing')}>
                    <i className="fas fa-question-circle"></i> Missing data
                </button>
                <button className={`tab`} onClick={() => setPage('automatedDashboard/overview')}>
                    <i className="fas fa-chart-line"></i> Dashboard
                </button>
            </div>
            
            <div className="content" style={{width: '100%'}}>
                {renderContent()}
            </div>
        </div>
    );
};

const DataPreview = ({ dataset }) => (
    <div>
        <h2>Data Preview</h2>
        <p style={{color: 'var(--text-secondary)'}}>First 5 rows of your dataset displayed in table format</p>
        <div className="table-container">
            <table>
                <thead>
                    <tr>{dataset.preview.length > 0 && Object.keys(dataset.preview[0]).map(key => <th key={key}>{key}</th>)}</tr>
                </thead>
                <tbody>
                    {dataset.preview.map((row, index) => (
                        <tr key={index}>
                            {Object.values(row).map((val, i) => <td key={i}>{val}</td>)}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);

const DataStructure = ({ dataset }) => {
    // Calculate statistics for each column
    const structure = dataset.columns_info.map(col => {
        const stats = calculateStats(dataset.preview, col.name, 
            col.dtype.includes('int') || col.dtype.includes('float') ? 'Numeric' : 'Text');
        
        return {
            name: col.name,
            type: col.dtype.includes('int') || col.dtype.includes('float') ? 'Numeric' : 
                  col.dtype.includes('datetime') ? 'Date' : 'Text',
            description: `Column with ${col.dtype} data type`,
            stats: stats.stats,
            details: stats.details
        };
    });

    // Count data types
    const typeCounts = {
        Numeric: structure.filter(col => col.type === 'Numeric').length,
        Text: structure.filter(col => col.type === 'Text').length,
        Date: structure.filter(col => col.type === 'Date').length
    };

    return (
        <div>
            <h2>Data Structure Analysis</h2>
            <p style={{color: 'var(--text-secondary)'}}>Column information, data types, and statistical overview</p>

            {structure.map(col => (
                <div key={col.name} className="structure-card">
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                        <div>
                            <h4 style={{margin: 0}}>{col.name} <span style={{fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 'normal'}}>{col.type}</span></h4>
                            <p style={{color: 'var(--text-secondary)', margin: '0.2rem 0 0 0'}}>{col.description}</p>
                        </div>
                        <div style={{textAlign: 'right'}}>
                            <p style={{margin: 0, color: 'var(--accent-green)'}}>{col.stats}</p>
                            <p style={{margin: 0, color: 'var(--text-secondary)'}}>{col.details}</p>
                        </div>
                    </div>
                </div>
            ))}
            
            <div className="structure-card">
                <h3>Data type Distribution</h3>
                 <div className="dist-cards">
                    <div className="dist-card numeric">
                        <h1 style={{margin: 0}}>{typeCounts.Numeric}</h1>
                        <p style={{margin: 0}}>Numeric Columns</p>
                    </div>
                    <div className="dist-card text">
                        <h1 style={{margin: 0}}>{typeCounts.Text}</h1>
                        <p style={{margin: 0}}>Text Columns</p>
                    </div>
                    <div className="dist-card date">
                        <h1 style={{margin: 0}}>{typeCounts.Date}</h1>
                        <p style={{margin: 0}}>Date Columns</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

const MissingData = ({ dataset }) => {
    // Calculate missing data
    const missingData = {};
    let totalMissing = 0;
    let totalCells = dataset.rows * dataset.columns;
    
    dataset.columns_info.forEach(col => {
        const missingValues = dataset.preview.filter(row => 
            row[col.name] === null || row[col.name] === undefined || row[col.name] === '').length;
        missingData[col.name] = missingValues;
        totalMissing += missingValues;
    });
    
    const completeness = ((totalCells - totalMissing) / totalCells) * 100;
    
    return (
         <div>
            <h2>Missing Data Analysis</h2>
            <p style={{color: 'var(--text-secondary)'}}>Comprehensive analysis of missing values across all columns</p>

            <div className="missing-data-card">
                <i className="fas fa-chart-line"></i>
                <h2 style={{color: 'var(--accent-green)'}}>{totalMissing === 0 ? 'Excellent Data Quality!' : 'Data Quality Check'}</h2>
                <p style={{color: 'var(--text-secondary)'}}>Your dataset has {totalMissing} missing values across all {dataset.columns} columns.</p>
                <div style={{display: 'flex', justifyContent: 'space-around', marginTop: '2rem'}}>
                    <div>
                        <h1 style={{margin: 0}}>{completeness.toFixed(1)}%</h1>
                        <p style={{margin: 0, color: 'var(--text-secondary)'}}>Data Completeness</p>
                    </div>
                    <div>
                        <h1 style={{margin: 0}}>{totalMissing}</h1>
                        <p style={{margin: 0, color: 'var(--text-secondary)'}}>Missing Values</p>
                    </div>
                </div>
            </div>
            
            <div className="structure-card">
                <h3>Missing Values by Column</h3>
                {Object.entries(missingData).map(([colName, missingCount]) => (
                    <div key={colName} style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                        <span style={{color: 'var(--text-primary)'}}>{colName}</span>
                        <span style={{color: missingCount > 0 ? 'var(--accent-pink)' : 'var(--accent-green)'}}>
                            {missingCount} missing ({((missingCount / dataset.rows) * 100).toFixed(1)}%)
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// UPDATED AUTOMATED DASHBOARD PAGE - This is the main component that works like Streamlit
const AutomatedDashboardPage = ({ setPage, dataset, initialTab }) => {
    const [activeTab, setActiveTab] = useState(initialTab);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [visualizations, setVisualizations] = useState(null);
    const [isLoadingViz, setIsLoadingViz] = useState(false);
    const [activeAnalysis, setActiveAnalysis] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [aiInsights, setAiInsights] = useState('');
    const [chatMessages, setChatMessages] = useState([]);
    const [chatInput, setChatInput] = useState('');

    // Analyze with AI - Matching Streamlit functionality
    const analyzeWithAI = async () => {
        setIsAnalyzing(true);
        try {
            const response = await fetch(`${BACKEND_URL}/api/analyze_with_ai`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    dataset_type: dataset.dataset_type
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setAnalysisResult(data.analysis_result);
            }
        } catch (error) {
            console.error('Error analyzing with AI:', error);
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Run specific analysis like Streamlit
    const runSpecificAnalysis = async (analysisName) => {
        setActiveAnalysis(analysisName);
        try {
            const response = await fetch(`${BACKEND_URL}/api/run_analysis_function`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    analysis_name: analysisName,
                    dataset_type: dataset.dataset_type
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setAnalysisData(data);
            }
        } catch (error) {
            console.error('Error running analysis:', error);
        } finally {
            setActiveAnalysis(null);
        }
    };

    // Generate visualizations like Streamlit
    const generateVisualizations = async () => {
        setIsLoadingViz(true);
        try {
            const response = await fetch(`${BACKEND_URL}/api/generate_visualizations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    theme: {
                        bg: "#0A071E",
                        text: "#ffffff",
                        primary: "#8a2be2",
                        secondary: "#1a183c",
                        chart_bg: "#1a183c",
                        grid: "#3a3863"
                    }
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setVisualizations(data.visualizations);
            }
        } catch (error) {
            console.error('Error generating visualizations:', error);
        } finally {
            setIsLoadingViz(false);
        }
    };

    // Get AI Insights like Streamlit
    const getAIInsights = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/get_ai_insights`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    dataset_type: dataset.dataset_type
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setAiInsights(data.insights);
            }
        } catch (error) {
            console.error('Error getting AI insights:', error);
        }
    };

    // Chat with data like Streamlit
    const sendChatMessage = async () => {
        if (!chatInput.trim()) return;

        const userMessage = { role: 'user', content: chatInput };
        setChatMessages(prev => [...prev, userMessage]);
        setChatInput('');

        try {
            const response = await fetch(`${BACKEND_URL}/api/chat_with_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: chatInput,
                    preview: dataset.preview,
                    history: chatMessages
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
            }
        } catch (error) {
            console.error('Error sending chat message:', error);
        }
    };

    // Export PDF like Streamlit
    const exportPDF = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/export_dashboard`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    format: 'pdf',
                    insights: aiInsights,
                    dataset_type: dataset.dataset_type
                }),
            });

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `autodash_report_${dataset.filename}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Export error:', error);
            alert('Error exporting PDF report');
        }
    };

    // Load data on component mount
    useEffect(() => {
        if (dataset) {
            generateVisualizations();
            getAIInsights();
        }
    }, [dataset]);

    useEffect(() => {
        if (activeTab === 'overview') {
            generateVisualizations();
        }
    }, [activeTab]);

    const renderContent = () => {
        switch(activeTab) {
            case 'overview': 
                return (
                    <OverviewTab 
                        dataset={dataset}
                        analysisResult={analysisResult}
                        isAnalyzing={isAnalyzing}
                        analyzeWithAI={analyzeWithAI}
                        runSpecificAnalysis={runSpecificAnalysis}
                        visualizations={visualizations}
                        isLoadingViz={isLoadingViz}
                        activeAnalysis={activeAnalysis}
                        analysisData={analysisData}
                        exportPDF={exportPDF}
                    />
                );
            case 'relationships': 
                return (
                    <RelationshipsTab 
                        dataset={dataset}
                        visualizations={visualizations}
                    />
                );
            case 'timeseries': 
                return (
                    <TimeSeriesTab 
                        dataset={dataset}
                    />
                );
            case 'ai-insights': 
                return (
                    <AIInsightsTab 
                        insights={aiInsights}
                    />
                );
            case 'chatbot': 
                return (
                    <ChatBotTab 
                        messages={chatMessages}
                        input={chatInput}
                        setInput={setChatInput}
                        sendMessage={sendChatMessage}
                    />
                );
            default: 
                return (
                    <OverviewTab 
                        dataset={dataset}
                        analysisResult={analysisResult}
                        isAnalyzing={isAnalyzing}
                        analyzeWithAI={analyzeWithAI}
                        runSpecificAnalysis={runSpecificAnalysis}
                        visualizations={visualizations}
                        isLoadingViz={isLoadingViz}
                        activeAnalysis={activeAnalysis}
                        analysisData={analysisData}
                        exportPDF={exportPDF}
                    />
                );
        }
    };

    return (
        <div className="page-wrapper" style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
            <BackButton onClick={() => setPage('datasetLoaded')} />
            <div style={{width: '100%'}}>
                <h2 style={{margin:0, color: '#FFFFFF'}}>Automated Dashboard</h2>
                <p style={{color: 'var(--text-secondary)'}}>Comprehensive analysis and visualization of your dataset with AI-powered insights</p>
            </div>
            
            {/* Streamlit-like Tabs */}
            <div className="dashboard-tabs">
                <button className={`tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
                    <i className="fas fa-eye"></i> Overview
                </button>
                <button className={`tab ${activeTab === 'relationships' ? 'active' : ''}`} onClick={() => setActiveTab('relationships')}>
                    <i className="fas fa-project-diagram"></i> Relationships
                </button>
                <button className={`tab ${activeTab === 'timeseries' ? 'active' : ''}`} onClick={() => setActiveTab('timeseries')}>
                    <i className="fas fa-chart-line"></i> Time Series
                </button>
                <button className={`tab ${activeTab === 'ai-insights' ? 'active' : ''}`} onClick={() => setActiveTab('ai-insights')}>
                    <i className="fas fa-robot"></i> AI Insights
                </button>
                <button className={`tab ${activeTab === 'chatbot' ? 'active' : ''}`} onClick={() => setActiveTab('chatbot')}>
                    <i className="fas fa-comments"></i> Chatbot
                </button>
            </div>

            <div className="content" style={{width: '100%'}}>
                {renderContent()}
            </div>
        </div>
    );
};

// UPDATED OVERVIEW TAB - Works exactly like Streamlit
const OverviewTab = ({ 
    dataset,
    analysisResult,
    isAnalyzing,
    analyzeWithAI,
    runSpecificAnalysis,
    visualizations,
    isLoadingViz,
    activeAnalysis,
    analysisData,
    exportPDF
}) => {
    // Get key metrics exactly like Streamlit
    const getKeyMetrics = () => {
        if (!dataset) return null;
        
        return {
            'Total Records': dataset.rows?.toLocaleString() || '0',
            'Total Features': dataset.columns?.toString() || '0',
            'Numeric Features': dataset.numeric_columns?.length || 0,
            'Categorical Features': dataset.categorical_columns?.length || 0
        };
    };

    const keyMetrics = getKeyMetrics();

    // Render visualization component
    const renderVisualization = (vizData, title, height = 400) => {
        if (!vizData) return null;
        
        try {
            const parsedData = typeof vizData === 'string' ? JSON.parse(vizData) : vizData;
            return (
                <div className="chart-container">
                    <div className="chart-title">{title}</div>
                    <div className="plotly-chart">
                        <Plot
                            data={parsedData.data}
                            layout={{...parsedData.layout, height}}
                            config={{
                                displayModeBar: true,
                                displaylogo: false,
                                responsive: true
                            }}
                        />
                    </div>
                </div>
            );
        } catch (error) {
            console.error('Error rendering visualization:', error);
            return null;
        }
    };

    return (
        <div>
            {/* AI Analysis Section - Exactly like Streamlit */}
            <div className="dashboard-card" style={{marginBottom: '2rem'}}>
                <h3>
                    <i className="fas fa-robot" style={{marginRight: '0.5rem', color: 'var(--accent-purple)'}}></i>
                    AI-Powered Analysis
                </h3>
                
                {!analysisResult ? (
                    <div style={{textAlign: 'center', padding: '2rem'}}>
                        <p style={{color: 'var(--text-secondary)'}}>
                            Get AI recommendations for the most relevant analyses based on your dataset
                        </p>
                        <button 
                            className="btn btn-gradient" 
                            onClick={analyzeWithAI}
                            disabled={isAnalyzing}
                        >
                            {isAnalyzing ? (
                                <>
                                    <i className="fas fa-spinner fa-spin" style={{marginRight: '0.5rem'}}></i>
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <i className="fas fa-magic" style={{marginRight: '0.5rem'}}></i>
                                    Analyze with AI
                                </>
                            )}
                        </button>
                    </div>
                ) : (
                    <div>
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
                            <p style={{color: 'var(--text-secondary)', margin: 0}}>
                                AI found {analysisResult.analysis_recommendations?.length || 0} relevant analyses
                                {analysisResult.confidence_score && (
                                    <span style={{marginLeft: '1rem', color: 'var(--accent-green)'}}>
                                        Confidence: {Math.round(analysisResult.confidence_score * 100)}%
                                    </span>
                                )}
                            </p>
                        </div>
                        
                        {/* Analysis Recommendations Grid like Streamlit */}
                        {analysisResult.analysis_recommendations && analysisResult.analysis_recommendations.length > 0 && (
                            <div className="analysis-recommendations-grid">
                                {analysisResult.analysis_recommendations.map((analysis, index) => (
                                    <div 
                                        key={index}
                                        className="analysis-recommendation-card"
                                        onClick={() => runSpecificAnalysis(analysis.name)}
                                    >
                                        <h4>{analysis.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
                                        <p style={{color: 'var(--text-secondary)', marginBottom: '0.5rem'}}>
                                            {analysis.description}
                                        </p>
                                        <small style={{color: 'var(--accent-blue)'}}>
                                            Columns: {analysis.columns?.join(', ')}
                                        </small>
                                        <div style={{marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                                            {activeAnalysis === analysis.name ? (
                                                <>
                                                    <i className="fas fa-spinner fa-spin" style={{color: 'var(--accent-orange)'}}></i>
                                                    <span style={{color: 'var(--accent-orange)'}}>Running Analysis...</span>
                                                </>
                                            ) : (
                                                <>
                                                    <i className="fas fa-play" style={{color: 'var(--accent-green)'}}></i>
                                                    <span style={{color: 'var(--accent-green)'}}>Run Analysis</span>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Key Metrics exactly like Streamlit */}
            {keyMetrics && (
                <div className="dashboard-card" style={{marginBottom: '2rem'}}>
                    <h3>📊 Key Metrics</h3>
                    <div className="streamlit-metrics">
                        {Object.entries(keyMetrics).map(([key, value]) => (
                            <div key={key} className="streamlit-metric">
                                <div className="streamlit-metric-value">{value}</div>
                                <div className="streamlit-metric-label">{key}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Visualizations Section */}
            <div className="dashboard-card" style={{marginBottom: '2rem'}}>
                <h3>Data Visualizations</h3>
                <p style={{color: 'var(--text-secondary)', marginBottom: '1.5rem'}}>
                    Interactive charts generated from your dataset
                </p>

                {isLoadingViz ? (
                    <div className="chart-loading">
                        <i className="fas fa-spinner fa-spin"></i>
                        <p>Generating visualizations...</p>
                    </div>
                ) : visualizations ? (
                    <div className="chart-grid">
                        {/* Render available visualizations */}
                        {Object.entries(visualizations).map(([key, vizData]) => {
                            if (key.includes('histogram') || key.includes('boxplot') || key.includes('barchart')) {
                                return (
                                    <div key={key}>
                                        {renderVisualization(vizData, key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))}
                                    </div>
                                );
                            }
                            return null;
                        })}
                        
                        {visualizations.correlation_matrix && 
                            renderVisualization(visualizations.correlation_matrix, 'Correlation Matrix')
                        }
                    </div>
                ) : (
                    <div className="chart-error">
                        <i className="fas fa-chart-line"></i>
                        <p>No visualizations available</p>
                    </div>
                )}
            </div>

            {/* PDF Export like Streamlit */}
            <div className="dashboard-card">
                <h3>Export Report</h3>
                <p style={{color: 'var(--text-secondary)', marginBottom: '1rem'}}>
                    Download a comprehensive PDF report of your analysis
                </p>
                <button className="btn btn-outline" onClick={exportPDF}>
                    <i className="fas fa-file-pdf"></i> Export PDF Report
                </button>
            </div>

            {/* Analysis Results */}
            {analysisData && (
                <div className="dashboard-card" style={{marginTop: '2rem'}}>
                    <h3>
                        <i className="fas fa-chart-line"></i> Analysis Results
                    </h3>
                    <div style={{
                        background: 'var(--primary-bg)',
                        padding: '1.5rem',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        marginTop: '1rem'
                    }}>
                        <h4 style={{color: 'var(--accent-green)', marginTop: 0}}>
                            {analysisData.analysis_name?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </h4>
                        
                        {analysisData.result && (
                            <div style={{color: 'var(--text-primary)'}}>
                                {analysisData.result.summary && (
                                    <div style={{marginBottom: '1rem'}}>
                                        <strong>Summary:</strong> {analysisData.result.summary}
                                    </div>
                                )}
                                
                                {analysisData.result.insights && analysisData.result.insights.length > 0 && (
                                    <div style={{marginBottom: '1rem'}}>
                                        <strong>Key Insights:</strong>
                                        <ul style={{paddingLeft: '1.5rem', margin: '0.5rem 0'}}>
                                            {analysisData.result.insights.map((insight, index) => (
                                                <li key={index}>{insight}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                                
                                {analysisData.result.recommendations && analysisData.result.recommendations.length > 0 && (
                                    <div>
                                        <strong>Recommendations:</strong>
                                        <ul style={{paddingLeft: '1.5rem', margin: '0.5rem 0'}}>
                                            {analysisData.result.recommendations.map((rec, index) => (
                                                <li key={index}>{rec}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// UPDATED TIME SERIES TAB - Works exactly like Streamlit
const TimeSeriesTab = ({ dataset }) => {
    const [dateCols, setDateCols] = useState([]);
    const [numCols, setNumCols] = useState([]);
    const [selectedDate, setSelectedDate] = useState("");
    const [selectedNum, setSelectedNum] = useState("");
    const [period, setPeriod] = useState(30);
    const [results, setResults] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // Load available columns
    useEffect(() => {
        fetchTimeSeriesColumns();
    }, [dataset]);

    const fetchTimeSeriesColumns = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/get_time_series_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setDateCols(data.date_cols || []);
                setNumCols(data.num_cols || []);
                if (data.date_cols?.length > 0) setSelectedDate(data.date_cols[0]);
                if (data.num_cols?.length > 0) setSelectedNum(data.num_cols[0]);
            }
        } catch (error) {
            console.error('Error fetching time series columns:', error);
        }
    };

    const runTimeSeriesAnalysis = async () => {
        if (!selectedDate || !selectedNum) return;
        
        setIsLoading(true);
        try {
            const response = await fetch(`${BACKEND_URL}/api/get_time_series_data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preview: dataset.preview,
                    date_col: selectedDate,
                    num_col: selectedNum,
                    period: period
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setResults(data.results);
            }
        } catch (error) {
            console.error('Error running time series analysis:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const renderVisualization = (vizData, title) => {
        if (!vizData) return null;
        
        try {
            const parsedData = typeof vizData === 'string' ? JSON.parse(vizData) : vizData;
            return (
                <div className="chart-container">
                    <div className="chart-title">{title}</div>
                    <div className="plotly-chart">
                        <Plot
                            data={parsedData.data}
                            layout={parsedData.layout}
                            config={{
                                displayModeBar: true,
                                displaylogo: false,
                                responsive: true
                            }}
                            style={{ width: '100%', height: '400px' }}
                        />
                    </div>
                </div>
            );
        } catch (error) {
            return <div>Error rendering chart</div>;
        }
    };

    return (
        <div className="dashboard-card">
            <h3>📈 Time Series Analysis</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: 0 }}>
                Analyze trends and patterns over time
            </p>

            {/* Column Selection like Streamlit */}
            <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "1rem", 
                marginBottom: "1.5rem",
                alignItems: "end"
            }}>
                <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Date Column
                    </label>
                    <select
                        value={selectedDate}
                        onChange={(e) => setSelectedDate(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '0.75rem',
                            borderRadius: '6px',
                            border: '1px solid var(--border-color)',
                            background: 'var(--primary-bg)',
                            color: 'var(--text-primary)'
                        }}
                    >
                        <option value="">Select Date Column</option>
                        {dateCols.map((col) => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Numeric Column
                    </label>
                    <select
                        value={selectedNum}
                        onChange={(e) => setSelectedNum(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '0.75rem',
                            borderRadius: '6px',
                            border: '1px solid var(--border-color)',
                            background: 'var(--primary-bg)',
                            color: 'var(--text-primary)'
                        }}
                    >
                        <option value="">Select Numeric Column</option>
                        {numCols.map((col) => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Period (days)
                    </label>
                    <input
                        type="number"
                        min="2"
                        max="365"
                        value={period}
                        onChange={(e) => setPeriod(Number(e.target.value))}
                        style={{
                            width: '100%',
                            padding: '0.75rem',
                            borderRadius: '6px',
                            border: '1px solid var(--border-color)',
                            background: 'var(--primary-bg)',
                            color: 'var(--text-primary)'
                        }}
                    />
                </div>

                <button 
                    className="btn btn-gradient" 
                    onClick={runTimeSeriesAnalysis}
                    disabled={isLoading || !selectedDate || !selectedNum}
                    style={{ height: 'fit-content' }}
                >
                    {isLoading ? (
                        <>
                            <i className="fas fa-spinner fa-spin"></i> Analyzing...
                        </>
                    ) : (
                        <>
                            <i className="fas fa-play"></i> Run Analysis
                        </>
                    )}
                </button>
            </div>

            {/* Results */}
            {results && (
                <div className="chart-grid">
                    {results.line_chart && 
                        renderVisualization(results.line_chart, `Time Series: ${selectedNum} over ${selectedDate}`)
                    }
                    
                    {results.decomposition && 
                        renderVisualization(results.decomposition, 'Time Series Decomposition')
                    }
                </div>
            )}

            {!dateCols.length && (
                <p style={{color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem'}}>
                    No datetime columns found for time series analysis.
                </p>
            )}
        </div>
    );
};

// UPDATED RELATIONSHIPS TAB
const RelationshipsTab = ({ dataset, visualizations }) => {
    const renderVisualization = (vizData, title) => {
        if (!vizData) return null;
        
        try {
            const parsedData = typeof vizData === 'string' ? JSON.parse(vizData) : vizData;
            return (
                <div className="chart-container">
                    <div className="chart-title">{title}</div>
                    <div className="plotly-chart">
                        <Plot
                            data={parsedData.data}
                            layout={parsedData.layout}
                            config={{
                                displayModeBar: true,
                                displaylogo: false,
                                responsive: true
                            }}
                            style={{ width: '100%', height: '400px' }}
                        />
                    </div>
                </div>
            );
        } catch (error) {
            return <div>Error rendering chart</div>;
        }
    };

    return (
        <div className="dashboard-card">
            <h3>🔗 Feature Relationships</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: 0 }}>
                Discover correlations and relationships between different features
            </p>

            {visualizations ? (
                <div className="chart-grid">
                    {visualizations.correlation_matrix && 
                        renderVisualization(visualizations.correlation_matrix, 'Correlation Matrix')
                    }
                    
                    {visualizations.scatter_matrix && 
                        renderVisualization(visualizations.scatter_matrix, 'Scatter Plot Matrix')
                    }
                </div>
            ) : (
                <div className="chart-error">
                    <i className="fas fa-exclamation-triangle"></i>
                    <p>No relationship data available</p>
                </div>
            )}
        </div>
    );
};

// UPDATED AI INSIGHTS TAB
const AIInsightsTab = ({ insights }) => {
    return (
        <div className="dashboard-card">
            <h3>🤖 AI Insights</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: 0 }}>
                Comprehensive summary and recommendations generated by AI
            </p>

            {insights ? (
                <div style={{
                    background: 'var(--primary-bg)',
                    padding: '1.5rem',
                    borderRadius: '8px',
                    marginTop: '1rem',
                    maxHeight: '500px',
                    overflowY: 'auto'
                }}>
                    <div style={{whiteSpace: 'pre-wrap', color: 'var(--text-secondary)', lineHeight: '1.6'}}>
                        {insights}
                    </div>
                </div>
            ) : (
                <div className="chart-error">
                    <i className="fas fa-exclamation-triangle"></i>
                    <p>No AI insights available. Run AI analysis first.</p>
                </div>
            )}
        </div>
    );
};

// UPDATED CHAT BOT TAB - Works exactly like Streamlit
const ChatBotTab = ({ messages, input, setInput, sendMessage }) => {
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="streamlit-chat-container">
            <h3>💬 Data Chatbot</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: 0 }}>
                Ask questions about your dataset and get instant answers
            </p>

            <div className="streamlit-chat-messages">
                {messages.map((message, index) => (
                    <div key={index} className={`streamlit-message ${message.role === 'user' ? 'streamlit-user-message' : 'streamlit-bot-message'}`}>
                        <p style={{margin: 0}}>{message.content}</p>
                    </div>
                ))}
            </div>

            <div style={{display: 'flex', gap: '0.5rem'}}>
                <input 
                    type="text" 
                    placeholder="Ask a question about your data..." 
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    style={{
                        flex: 1,
                        padding: '0.8rem',
                        border: '1px solid var(--border-color)',
                        borderRadius: '8px',
                        background: 'var(--primary-bg)',
                        color: 'var(--text-primary)'
                    }}
                />
                <button 
                    onClick={sendMessage}
                    disabled={!input.trim()}
                    className="btn btn-gradient"
                    style={{padding: '0.8rem 1.5rem'}}
                >
                    <i className="fas fa-paper-plane"></i>
                </button>
            </div>

            <div style={{marginTop: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap'}}>
                <span style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>Try asking:</span>
                <button 
                    type="button" 
                    onClick={() => setInput("What are the main trends in this data?")}
                    style={{
                        background: 'none',
                        border: '1px solid var(--border-color)',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        color: 'var(--text-secondary)',
                        fontSize: '0.8rem',
                        cursor: 'pointer'
                    }}
                >
                    Trends
                </button>
                <button 
                    type="button" 
                    onClick={() => setInput("Show me the correlations between numeric columns")}
                    style={{
                        background: 'none',
                        border: '1px solid var(--border-color)',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        color: 'var(--text-secondary)',
                        fontSize: '0.8rem',
                        cursor: 'pointer'
                    }}
                >
                    Correlations
                </button>
                <button 
                    type="button" 
                    onClick={() => setInput("What insights can you provide about this dataset?")}
                    style={{
                        background: 'none',
                        border: '1px solid var(--border-color)',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        color: 'var(--text-secondary)',
                        fontSize: '0.8rem',
                        cursor: 'pointer'
                    }}
                >
                    Insights
                </button>
            </div>
        </div>
    );
};

// UPDATED AI BOT ASSISTANT - Works exactly like Streamlit
const AIBotAssistant = ({ setPage, previousPage }) => {
    const [messages, setMessages] = useState([
        { 
            role: 'assistant', 
            content: "Hello! I'm your AI data assistant. How can I help you with your data analysis today?" 
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/api/chat_with_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input,
                    preview: [], // You might want to pass dataset here if available
                    history: messages
                }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
            } else {
                setMessages(prev => [...prev, { 
                    role: 'assistant', 
                    content: "I'm sorry, I'm having trouble processing your request. Please try again." 
                }]);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: "I'm sorry, I encountered a connection error. Please check your connection and try again." 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="page-wrapper">
            <BackButton onClick={() => setPage(previousPage)} text="Back to Dashboard"/>
            
            <div className="bot-container">
                <div className="bot-header">
                    <i className="fas fa-robot"></i>
                    <h3>AI Data Assistant</h3>
                </div>
                
                <div className="bot-messages">
                    {messages.map((message, index) => (
                        <div key={index} className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'}`}>
                            <p>{message.content}</p>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="message bot-message">
                            <p><i className="fas fa-spinner fa-spin"></i> Thinking...</p>
                        </div>
                    )}
                </div>
                
                <form className="bot-input-form" onSubmit={handleSendMessage}>
                    <input 
                        type="text" 
                        placeholder="Ask a question about data analysis..." 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={isLoading}
                    />
                    <button type="submit" disabled={isLoading}>
                        <i className="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
        </div>
    );
};

const AboutUsPage = ({ setPage }) => {
    return (
        <div className="page-wrapper">
            <BackButton onClick={() => setPage('home')} text="Back to Home"/>
            <div className="about-page">
                <div style={{marginTop: '6rem'}}>
                    <i className="fas fa-brain brain-icon"></i>
                    <h1>About AutoDash AI</h1>
                    <p style={{color: 'var(--text-secondary)', fontSize: '1.2rem'}}>
                        We're revolutionizing data analysis by making advanced analytics accessible to everyone, regardless of technical expertise.
                    </p>
                </div>
                <div className="about-cards">
                    <div className="about-card">
                        <h3>Our Mission</h3>
                        <p style={{color: 'var(--text-secondary)'}}>To democratize data analytics and empower every individual and organization to make intelligent, data-driven decisions with ease.</p>
                    </div>
                    <div className="about-card">
                        <h3>Our Vision</h3>
                        <p style={{color: 'var(--text-secondary)'}}>A passionate team of data scientists, AI engineers, and UX designers committed to creating exceptional analytical experiences.</p>
                    </div>
                    <div className="about-card">
                        <h3>Our Team</h3>
                        <p style={{color: 'var(--text-secondary)'}}>A world where data insights are instantly accessible, beautifully visualized, and actionable for businesses of all sizes.</p>
                    </div>
                </div>
                <div style={{marginTop: '3rem'}}>
                    <h2>Ready to Transform Your Data?</h2>
                    <p style={{color: 'var(--text-secondary)'}}>Join thousands of users who trust AutoDash AI for their data analysis needs.</p>
                    <button className="btn btn-secondary mt-1" onClick={() => setPage('upload')}>Start Analyzing Today</button>
                </div>
            </div>
        </div>
    );
};

const BotFab = ({ setPage, setPreviousPage, currentPage }) => {
    const handleClick = () => {
        setPreviousPage(currentPage);
        setPage('bot');
    };
    return (
        <div className="bot-fab" onClick={handleClick}>
            <i className="fas fa-robot"></i>
        </div>
    );
};

// --- MAIN APP COMPONENT ---

let AppInstance;

function App() {
    const [page, setPage] = useState('home');
    const [previousPage, setPreviousPage] = useState('home');
    const [isLoggedIn, setLoggedIn] = useState(false);
    const [file, setFile] = useState(null);
    const [dataset, setDataset] = useState(null);

    AppInstance = { setPage, setPreviousPage, setLoggedIn, setFile, setDataset };

    // This injects the styles into the document's head
    useEffect(() => {
        const styleElement = document.createElement('style');
        styleElement.innerHTML = AppStyles;
        document.head.appendChild(styleElement);
        return () => {
            if (document.head.contains(styleElement)) {
                document.head.removeChild(styleElement);
            }
        };
    }, []);

    const renderPage = () => {
        const explorerMatch = page.match(/^dataExplorer\/(.*)$/);
        if (explorerMatch && dataset) {
            return <DataExplorerPage setPage={setPage} dataset={dataset} initialTab={explorerMatch[1]} />;
        }
        
        const dashboardMatch = page.match(/^automatedDashboard\/(.*)$/);
        if (dashboardMatch && dataset) {
            return <AutomatedDashboardPage setPage={setPage} dataset={dataset} initialTab={dashboardMatch[1]} />;
        }
        
        switch (page) {
            case 'home':
                return <HomePage setPage={setPage} />;
            case 'about':
                return <AboutUsPage setPage={setPage} />;
            case 'bot':
                return <AIBotAssistant setPage={setPage} previousPage={previousPage} />;
            case 'login':
                return <LoginPage setPage={setPage} setLoggedIn={setLoggedIn} />;
            case 'signup':
                return <SignUpPage setPage={setPage} setLoggedIn={setLoggedIn} />;
            case 'upload':
                if (file) {
                    return <FileUploadedView setPage={setPage} file={file} setDataset={setDataset} setFile={setFile} />;
                }
                return <UploadDatasetPage setPage={setPage} isLoggedIn={isLoggedIn} setFile={setFile} />;
            case 'datasetLoaded':
                 if (dataset) {
                    return <DatasetLoadedPage setPage={setPage} dataset={dataset} />;
                }
                setPage('upload');
                return null;
            default:
                return <HomePage setPage={setPage} />;
        }
    };

    return (
        <div className="app-container">
            {page !== 'bot' && (
                <Header 
                    setPage={setPage} 
                    isLoggedIn={isLoggedIn} 
                    setLoggedIn={setLoggedIn}
                />
            )}
            <main className="main-content">
                {renderPage()}
            </main>
            {page !== 'bot' && <BotFab setPage={setPage} setPreviousPage={setPreviousPage} currentPage={page}/>}
        </div>
    );
}

// This is a workaround for state management in this single-file setup
const setPage = (page) => AppInstance.setPage(page);
const setPreviousPage = (page) => AppInstance.setPreviousPage(page);
const setLoggedIn = (status) => AppInstance.setLoggedIn(status);
const setFile = (file) => AppInstance.setFile(file);
const setDataset = (data) => AppInstance.setDataset(data);

export default App;