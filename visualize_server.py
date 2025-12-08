#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型长文本能力评估可视化工具
单文件Flask应用 - 支持表格、图表展示和case详情查看
"""

from flask import Flask, render_template_string, jsonify, request
import json
import os
from pathlib import Path
from collections import defaultdict
import re

app = Flask(__name__)

# 数据集到指标的映射
DATASET_TO_METRICS = {
    "json_kv": "substring_exact_match",
    "json_kv_chinese_poem": "substring_exact_match",
    "json_kv_chinese_poem_balanced": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    "narrativeqa": "gpt-4-score",
    "msmarco_rerank_psg": "NDCG@10",
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    "qmsum": "rougeL_recall",
    "multi_lexsum": "gpt-4-f1",
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    "infbench_qa": "rougeL_f1",
    "infbench_choice": "exact_match",
    "infbench_sum": "gpt-4-f1",
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

# 复合指标定义
CUSTOM_AVGS = {
    "Recall": ["json_kv_chinese_poem_balanced substring_exact_match", "json_kv_chinese_poem substring_exact_match", "json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", 
               "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 
            'clinic150 exact_match', 'nlu exact_match'],
}


def extract_task_name(filename):
    """从文件名中提取任务名"""
    name = filename.replace('.json.score', '').replace('.json', '')
    
    # 匹配模式
    patterns = [
        r'icl_(\w+)_\d+shot',  # ICL任务
        r'(ruler_\w+)_eval',    # RULER任务
        r'(json_kv_chinese_poem_balanced)_eval', # JSON_KV拓展，需要放在JSON_KV通配符前面
        r'(json_kv_chinese_poem)_eval', # JSON_KV拓展，需要放在JSON_KV通配符前面
        r'(json_kv)_eval',      # JSON KV任务
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            task = match.group(1)
            return task
    
    parts = name.split('_')
    if len(parts) >= 2:
        if parts[0] == 'icl':
            return parts[1]
        elif parts[0] == 'ruler':
            return '_'.join(parts[:3]) if len(parts) >= 3 else '_'.join(parts[:2])
        elif parts[0] == 'json':
            if "chinese_poem" in name:
                print("FOUND JSON_KV_CHINESE_POEM")
                if "balanced" in name:
                    return "json_kv_chinese_poem_balanced"
                else:
                    return "json_kv_chinese_poem"
            else:
                return 'json_kv'
    
    return name.split('_')[0] if '_' in name else name


def extract_seqlen(filename):
    """从文件名中提取序列长度"""
    match = re.search(r'in(\d+)', filename)
    if match:
        return int(match.group(1))
    return 32768


def load_score_file(filepath):
    """加载score文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_metric_value(score_data, task_name):
    """根据任务名获取对应的指标值"""
    if not score_data:
        return None
    
    metrics = DATASET_TO_METRICS.get(task_name)
    
    if metrics is None:
        return None
    
    if isinstance(metrics, list):
        for metric in metrics:
            if metric in score_data:
                return score_data[metric]
        return None
    else:
        return score_data.get(metrics)


def scan_directory(base_path='output'):
    """扫描目录结构，提取所有模型的评估结果"""
    results = defaultdict(lambda: defaultdict(dict))
    
    base = Path(base_path)
    if not base.exists():
        return results
    
    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        for score_file in model_dir.glob('*.json.score'):
            filename = score_file.name
            task_name = extract_task_name(filename)
            seqlen = extract_seqlen(filename)
            
            score_data = load_score_file(score_file)
            metric_value = get_metric_value(score_data, task_name)
            
            if metric_value is not None:
                results[model_name][seqlen][f"{task_name}"] = {
                    'score': metric_value,
                    'metric': DATASET_TO_METRICS.get(task_name),
                    'filename': filename
                }
    
    return results


def calculate_custom_metrics(model_results):
    """计算复合指标（Recall和ICL）"""
    for model_name, seqlen_data in model_results.items():
        for seqlen, metrics in seqlen_data.items():
            for custom_name, task_metrics in CUSTOM_AVGS.items():
                scores = []
                for tm in task_metrics:
                    task, metric = tm.split(' ', 1)
                    if task in metrics:
                        scores.append(metrics[task]['score'])
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    metrics[custom_name] = {
                        'score': avg_score,
                        'metric': 'average',
                        'filename': 'computed'
                    }
    
    return model_results


def load_case_data(model_name, filename, base_path='output'):
    # return {
    #         'meta': "none",
    #         'cases': "none",
    #         'total_sample': 100,
    #         'valid_ratio': 0.97,
    #     }

    print("[DEBUG] load_case_data", model_name, filename, base_path)
    """加载具体的case数据"""
    json_file = filename.replace('.json.score', '.json')
    find = False
    for i in os.listdir(base_path):
        if model_name in i:
            model_name = i
            find = True
            break
    if not find:
        print("[DEBUG] load_case_data not find model_name", model_name)
        return None

    filepath = Path(base_path) / model_name / json_file
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        meta = raw_data.get('args', {})
        
        task_name = extract_task_name(filename)
        final_metric = DATASET_TO_METRICS.get(task_name)
        if isinstance(final_metric, list):
            final_metric = final_metric[0]
        
        cases = []
        data_list = raw_data.get('data', [])
        
        for idx, item in enumerate(data_list):
            case = {
                'id': item.get('index', item.get('example', item.get('id', idx))),
                'question': item.get('question', item.get('query', '')),
                'input_text': item.get('input_text', item.get('input', '')),
                'output': item.get('output', ''),
                'parsed_output': item.get('parsed_output', item.get('output', '')),
                'answer': item.get('answer', item.get('label', '')),
                'input_len': item.get('input_len', 0),
                'output_len': item.get('output_len', 0),
            }
            
            is_correct = False
            if final_metric and final_metric in item:
                value = item[final_metric]
                is_correct = (value == 1.0 or value == 1 or value is True)
            elif 'exact_match' in item:
                is_correct = (item['exact_match'] == 1.0 or item['exact_match'] == 1)
            elif 'substring_exact_match' in item:
                is_correct = (item['substring_exact_match'] == 1.0 or item['substring_exact_match'] == 1)
            elif 'ruler_recall' in item:
                is_correct = (item['ruler_recall'] == 1.0 or item['ruler_recall'] == 1)
            
            case['is_correct'] = is_correct
            cases.append(case)
        
        cases.sort(key=lambda x: (x['is_correct'], x['id']))
        
        return {
            'meta': meta,
            'cases': cases,
            'total_sample': raw_data.get('total_sample', len(cases)),
            'valid_ratio': raw_data.get('valid_ratio', "100.0%")
        }
    
    except Exception as e:
        print(f"Error loading case data from {filepath}: {e}")
        return None


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型长文本能力评估可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #2c3e50; margin-bottom: 30px; text-align: center; font-size: 28px; }
        .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .control-group { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        label { font-weight: 600; color: #555; }
        select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; background: white; font-size: 14px; cursor: pointer; }
        select:hover { border-color: #4CAF50; }
        .view-tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; background: white; border: 2px solid #ddd; border-radius: 6px; cursor: pointer; font-weight: 600; transition: all 0.3s; }
        .tab-btn:hover { border-color: #4CAF50; }
        .tab-btn.active { background: #4CAF50; color: white; border-color: #4CAF50; }
        .view-content { display: none; }
        .view-content.active { display: block; }
        .table-container { background: white; border-radius: 8px;  padding: 20px 20px 20px 0; /* 左 padding = 0，避免 sticky 偏移 */; overflow-x: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; min-width: 800px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        /* 🛠 固定第一列（指标列） */
        #results-table th:first-child,
        #results-table td:first-child {
            position: sticky;
            left: 0;
            background: white;   /* 避免滚动时透明 */
            z-index: 5;          /* 保证在其他列之上 */
        }
        /* 🛠 固定第一行 + 第一列（表头“指标”） */
        #results-table thead th:first-child {
            position: sticky;
            top: 0;
            left: 0;
            background: white;
            z-index: 20;   /* 必须比普通 sticky 列更高 */
        }
        /* 渲染分割线 */
        .metric-divider td {
            background: #fafafa;
            padding: 6px 12px;
        }
        .divider-line {
            border: none;
            border-top: 2px solid #ddd;
            margin-top: 6px;
        }

        .divider-label {
            font-weight: 700;
            color: #666;
            margin-bottom: 2px;
        }

        th { background: #f8f9fa; font-weight: 600; color: #555; position: sticky; top: 0; z-index: 10; }
        tr:hover { background: #f8f9fa; }
        .metric-name { font-weight: 600; color: #2c3e50; }
        .score-cell { font-family: 'Monaco', 'Consolas', monospace; font-weight: 600; }
        .score-high { color: #27ae60; }
        .score-medium { color: #f39c12; }
        .score-low { color: #e74c3c; }
        .chart-container { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        canvas { max-height: 500px; }
        .case-view { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .meta-info { background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
        .meta-row { display: flex; gap: 30px; flex-wrap: wrap; margin-bottom: 8px; }
        .meta-item { display: flex; gap: 8px; }
        .meta-label { font-weight: 600; color: #555; }
        .meta-value { color: #333; }
        .case-controls { display: flex; gap: 15px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }
        .filter-btn { padding: 8px 16px; border: 2px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-weight: 500; transition: all 0.3s; }
        .filter-btn:hover { border-color: #4CAF50; }
        .filter-btn.active { background: #4CAF50; color: white; border-color: #4CAF50; }
        .case-table-container { max-height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; }
        .case-table { width: 100%; border-collapse: collapse; }
        .case-table th { background: #2c3e50; color: white; padding: 12px; position: sticky; top: 0; z-index: 10; }
        .case-table td { padding: 12px; border-bottom: 1px solid #eee; vertical-align: top; }
        .case-table tr:hover { background: #f0f8ff; }
        .case-correct { background: #d4edda !important; }
        .case-incorrect { background: #f8d7da !important; }
        .case-text { max-width: 400px; max-height: 100px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .badge-correct { background: #27ae60; color: white; }
        .badge-incorrect { background: #e74c3c; color: white; }
        .loading { text-align: center; padding: 40px; color: #999; }
        .error { background: #fee; color: #c33; padding: 15px; border-radius: 6px; margin: 20px 0; }
        #results-table td,
        #results-table th {
            white-space: nowrap;      /* 不自动换行 */
            word-break: keep-all;     /* 保留完整单词，不在连字符断开 */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 模型长文本能力评估可视化</h1>
        
        <div class="controls">
            <div class="control-group">
                <div>
                    <label for="seqlen-select">序列长度:</label>
                    <select id="seqlen-select" onchange="updateView()">
                        <option value="">加载中...</option>
                    </select>
                </div>
                <div>
                    <label for="model-select">模型:</label>
                    <select id="model-select" onchange="loadCaseData()">
                        <option value="">选择模型</option>
                    </select>
                </div>
                <div>
                    <label for="task-select">任务:</label>
                    <select id="task-select" onchange="loadCaseData()">
                        <option value="">选择任务</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="view-tabs">
            <button class="tab-btn active" onclick="switchView('table')">📊 表格视图</button>
            <button class="tab-btn" onclick="switchView('chart')">📈 图表视图</button>
            <button class="tab-btn" onclick="switchView('case')">🔎 Case详情</button>
        </div>
        
        <div id="table-view" class="view-content active">
            <div class="table-container">
                <table id="results-table">
                    <thead><tr><th>指标</th></tr></thead>
                    <tbody><tr><td class="loading">正在加载数据...</td></tr></tbody>
                </table>
            </div>
        </div>
        
        <div id="chart-view" class="view-content">
            <div class="chart-container">
                <canvas id="comparison-chart"></canvas>
            </div>
        </div>
        
        <div id="case-view" class="view-content">
            <div class="case-view">
                <div id="case-content">
                    <p class="loading">请选择模型和任务查看详细case</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let allData = {};
        let currentSeqlen = null;
        let chartInstance = null;
        
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                allData = await response.json();
                
                const seqlens = new Set();
                Object.values(allData).forEach(modelData => {
                    Object.keys(modelData).forEach(seqlen => seqlens.add(parseInt(seqlen)));
                });
                
                const seqlenSelect = document.getElementById('seqlen-select');
                seqlenSelect.innerHTML = '';
                [...seqlens].sort((a, b) => a - b).forEach(seqlen => {
                    const option = document.createElement('option');
                    option.value = seqlen;
                    option.textContent = seqlen;
                    seqlenSelect.appendChild(option);
                });
                
                if (seqlens.size > 0) {
                    currentSeqlen = Math.max(...seqlens);
                    seqlenSelect.value = currentSeqlen;
                    updateView();
                }
                
                const modelSelect = document.getElementById('model-select');
                modelSelect.innerHTML = '<option value="">选择模型</option>';
                Object.keys(allData).forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.querySelector('#results-table tbody').innerHTML = 
                    '<tr><td class="error">数据加载失败: ' + error.message + '</td></tr>';
            }
        }
        
        function updateView() {
            currentSeqlen = document.getElementById('seqlen-select').value;
            if (!currentSeqlen) return;
            
            const activeView = document.querySelector('.view-content.active').id;
            if (activeView === 'table-view') {
                updateTable();
            } else if (activeView === 'chart-view') {
                updateChart();
            }
        }
        
        function updateTable() {
            if (!currentSeqlen) return;

            // 1. 收集当前 seq_len 下所有出现过的指标
            const metrics = new Set();
            Object.values(allData).forEach(modelData => {
                const seqlenData = modelData[currentSeqlen];
                if (seqlenData) {
                    Object.keys(seqlenData).forEach(metric => metrics.add(metric));
                }
            });

            // 如果一个模型在当前 seq_len 下完全没数据，就不展示这一列
            const models = Object.keys(allData);
            const filteredModels = models.filter(model => {
                const data = allData[model][currentSeqlen];
                return data && Object.keys(data).length > 0;
            });

            // 如果没有任何模型有数据，给个提示
            if (filteredModels.length === 0) {
                document.querySelector('#results-table thead').innerHTML = '<tr><th>指标</th></tr>';
                document.querySelector('#results-table tbody').innerHTML =
                    '<tr><td class="loading">当前序列长度没有可用数据</td></tr>';
                return;
            }

            // 2. 构建表头（使用过滤后的模型列表）
            let headerHTML = '<tr><th>指标</th>';
            filteredModels.forEach(model => {
                headerHTML += `<th>${model}</th>`;
            });
            headerHTML += '</tr>';

            // 3. 指标分组定义
            const groupOrder = {
                summary: ["ICL", "Recall"], // 汇总指标
                ICL_items: ["trec_coarse", "trec_fine", "banking77", "clinic150", "nlu"],
                Recall_items: ["json_kv", "json_kv_chinese_poem", "json_kv_chinese_poem_balanced", "ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv"],
            };

            // 4. 根据分组把当前 metrics 分桶
            let summaryMetrics = [];
            let ICLMetrics = [];
            let RecallMetrics = [];
            let otherMetrics = [];

            metrics.forEach(metric => {
                if (groupOrder.summary.includes(metric)) {
                    summaryMetrics.push(metric);
                } else if (groupOrder.ICL_items.includes(metric)) {
                    ICLMetrics.push(metric);
                } else if (groupOrder.Recall_items.includes(metric)) {
                    RecallMetrics.push(metric);
                } else {
                    otherMetrics.push(metric);
                }
            });

            // 5. 行渲染函数：渲染一行指标
            function renderMetricRow(metric) {
                let row = '<tr>';
                row += `<td class="metric-name">${metric}</td>`;
                filteredModels.forEach(model => {
                    const data = allData[model][currentSeqlen];
                    if (data && data[metric]) {
                        const score = data[metric].score;
                        const scoreClass =
                            score >= 80 ? 'score-high' :
                            score >= 50 ? 'score-medium' : 'score-low';
                        row += `<td class="score-cell ${scoreClass}">${score.toFixed(2)}</td>`;
                    } else {
                        row += '<td>-</td>';
                    }
                });
                row += '</tr>';
                return row;
            }

            // 6. 分割线渲染函数
            function renderDivider(text) {
                return `
                    <tr class="metric-divider">
                        <td class="group-label-cell">
                            <div class="divider-label">${text}</div>
                        </td>
                        <td colspan="${filteredModels.length}">
                            <hr class="divider-line"/>
                        </td>
                    </tr>`;
            }

            // 7. 按顺序拼接 bodyHTML
            let bodyHTML = "";

            // 第一块：汇总指标 ICL / Recall
            summaryMetrics.forEach(metric => {
                bodyHTML += renderMetricRow(metric);
            });
            if (summaryMetrics.length > 0 && (ICLMetrics.length > 0 || RecallMetrics.length > 0)) {
                bodyHTML += renderDivider("ICL 细项指标");
            }

            // 第二块：ICL 细项
            ICLMetrics.forEach(metric => {
                bodyHTML += renderMetricRow(metric);
            });
            if (ICLMetrics.length > 0 && RecallMetrics.length > 0) {
                bodyHTML += renderDivider("Recall 细项指标");
            }

            // 第三块：Recall 细项
            RecallMetrics.forEach(metric => {
                bodyHTML += renderMetricRow(metric);
            });

            // 其他未分组指标（如果有的话，排在最后，按名字排序）
            otherMetrics.sort().forEach(metric => {
                bodyHTML += renderMetricRow(metric);
            });

            // 8. 写回 DOM
            document.querySelector('#results-table thead').innerHTML = headerHTML;
            document.querySelector('#results-table tbody').innerHTML = bodyHTML;
        }
        
        function updateChart() {
            if (!currentSeqlen) return;
            
            const ctx = document.getElementById('comparison-chart');
            
            const metrics = new Set();
            Object.values(allData).forEach(modelData => {
                if (modelData[currentSeqlen]) {
                    Object.keys(modelData[currentSeqlen]).forEach(metric => metrics.add(metric));
                }
            });
            
            const metricsList = [...metrics].sort();
            const models = Object.keys(allData);
            
            const datasets = models.map((model, idx) => {
                const colors = [
                    '#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
                    '#00BCD4', '#CDDC39', '#FF5722', '#795548', '#607D8B'
                ];
                const color = colors[idx % colors.length];
                
                const data = metricsList.map(metric => {
                    const modelData = allData[model][currentSeqlen];
                    return modelData && modelData[metric] ? modelData[metric].score : 0;
                });
                
                return {
                    label: model,
                    data: data,
                    backgroundColor: color + '80',
                    borderColor: color,
                    borderWidth: 2
                };
            });
            
            if (chartInstance) {
                chartInstance.destroy();
            }
            
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: metricsList,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `模型对比 - Seq Length: ${currentSeqlen}`,
                            font: { size: 18 }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: '分数'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '指标'
                            }
                        }
                    }
                }
            });
        }
        
        function switchView(view) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.view-content').forEach(content => content.classList.remove('active'));
            document.getElementById(view + '-view').classList.add('active');
            
            if (view === 'chart') {
                setTimeout(updateChart, 100);
            }
        }
        
        async function loadCaseData() {
            const model = document.getElementById('model-select').value;
            const task = document.getElementById('task-select').value;
            
            if (!model || !task) return;
            
            const caseContent = document.getElementById('case-content');
            caseContent.innerHTML = '<p class="loading">加载中...</p>';
            
            try {
                const response = await fetch(`/api/case?model=${model}&task=${task}`);
                const data = await response.json();
                
                if (data.error) {
                    caseContent.innerHTML = `<p class="error">${data.error}</p>`;
                    return;
                }
                
                renderCaseData(data);
            } catch (error) {
                caseContent.innerHTML = `<p class="error">加载失败: ${error.message}</p>`;
            }
        }
        
        function renderCaseData(data) {
            const caseContent = document.getElementById('case-content');
            
            let html = '<div class="meta-info">';
            html += '<h3>📋 评估信息</h3>';
            html += '<div class="meta-row">';
            html += `<div class="meta-item"><span class="meta-label">数据集:</span><span class="meta-value">${data.meta.datasets || 'N/A'}</span></div>`;
            html += `<div class="meta-item"><span class="meta-label">最大测试样本:</span><span class="meta-value">${data.meta.max_test_samples || 'N/A'}</span></div>`;
            html += `<div class="meta-item"><span class="meta-label">Shots:</span><span class="meta-value">${data.meta.shots || 0}</span></div>`;
            html += '</div>';
            html += '<div class="meta-row">';
            html += `<div class="meta-item"><span class="meta-label">输入最大长度:</span><span class="meta-value">${data.meta.input_max_length || 'N/A'}</span></div>`;
            html += `<div class="meta-item"><span class="meta-label">Temperature:</span><span class="meta-value">${data.meta.temperature || 0}</span></div>`;
            html += `<div class="meta-item"><span class="meta-label">Top P:</span><span class="meta-value">${data.meta.top_p || 1.0}</span></div>`;
            html += '</div>';
            html += '<div class="meta-row">';
            html += `<div class="meta-item"><span class="meta-label">总样本数:</span><span class="meta-value">${data.total_sample}</span></div>`;
            html += `<div class="meta-item"><span class="meta-label">有效率:</span><span class="meta-value">${data.valid_ratio }</span></div>`;
            html += '</div>';
            html += '</div>';
            
            html += '<div class="case-controls">';
            html += '<button class="filter-btn active" onclick="filterCases(\\'all\\')">全部</button>'
            html += '<button class="filter-btn" onclick="filterCases(\\'incorrect\\')">仅错误</button>'
            html += '<button class="filter-btn" onclick="filterCases(\\'correct\\')">仅正确</button>'
            html += '</div>';
            
            html += '<div class="case-table-container">';
            html += '<table class="case-table">';
            html += '<thead><tr>';
            html += '<th>ID</th><th>正确性</th><th>Question</th><th>模型输入</th><th>Answer</th><th>Output</th><th>Input Len</th><th>Output Len</th>';
            html += '</tr></thead>';
            html += '<tbody>';
            
            data.cases.forEach(c => {
                const rowClass = c.is_correct ? 'case-correct' : 'case-incorrect';
                const badge = c.is_correct ? 
                    '<span class="badge badge-correct">✓ 正确</span>' : 
                    '<span class="badge badge-incorrect">✗ 错误</span>';
                
                html += `<tr class="${rowClass}" data-correct="${c.is_correct}">`;
                html += `<td>${c.id}</td>`;
                html += `<td>${badge}</td>`;
                html += `<td><div class="case-text">${escapeHtml(c.question || '')}</div></td>`;
                html += `<td><div class="case-text">${escapeHtml(c.input_text || '')}</div></td>`;
                html += `<td><div class="case-text">${escapeHtml(String(c.answer || ''))}</div></td>`;
                html += `<td><div class="case-text">${escapeHtml(c.parsed_output || c.output || '')}</div></td>`;
                html += `<td>${c.input_len}</td>`;
                html += `<td>${c.output_len}</td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table></div>';
            
            caseContent.innerHTML = html;
        }
        
        function filterCases(filter) {
            document.querySelectorAll('.case-controls .filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            const rows = document.querySelectorAll('.case-table tbody tr');
            rows.forEach(row => {
                if (filter === 'all') {
                    row.style.display = '';
                } else if (filter === 'correct') {
                    row.style.display = row.dataset.correct === 'true' ? '' : 'none';
                } else if (filter === 'incorrect') {
                    row.style.display = row.dataset.correct === 'false' ? '' : 'none';
                }
            });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('model-select').addEventListener('change', function() {
            const model = this.value;
            const taskSelect = document.getElementById('task-select');
            taskSelect.innerHTML = '<option value="">选择任务</option>';
            
            if (model && allData[model] && currentSeqlen) {
                const tasks = Object.keys(allData[model][currentSeqlen] || {});
                tasks.forEach(task => {
                    const taskInfo = allData[model][currentSeqlen][task];
                    // 🚫 如果是复合指标（metric 为 "average" 或 filename 为 "computed"），跳过
                    if (taskInfo.metric === 'average' || taskInfo.filename === 'computed') return;
                    const option = document.createElement('option');
                    option.value = allData[model][currentSeqlen][task].filename;
                    option.textContent = task;
                    taskSelect.appendChild(option);
                });
            }
        });
        
        window.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


# 在 /api/data 接口和关键函数中添加诊断输出，方便定位加载问题。
@app.route('/api/data')
def get_data():
    print("[DEBUG] /api/data 被调用")
    results = scan_directory()
    print(f"[DEBUG] 扫描结果模型数量: {len(results)}")
    for model_name, data in results.items():
        print(f"  [DEBUG] 模型: {model_name}, 序列长度数量: {len(data)}")
        for seqlen, tasks in data.items():
            print(f"    [DEBUG] 序列长度: {seqlen}, 任务数量: {len(tasks)}")
    
    results = calculate_custom_metrics(results)
    print("[DEBUG] 计算复合指标完成")
    return jsonify(results)

@app.route('/api/case')
def api_case():
    model_name = request.args.get('model')
    filename = request.args.get('task')

    if not model_name or not filename:
        return jsonify({'error': '缺少参数 model 或 task'}), 400

    data = load_case_data(model_name, filename)
    if not data:
        return jsonify({'error': f'无法加载 {model_name}/{filename} 的case数据'}), 404

    return jsonify(data)

def scan_directory(base_path='output'):
    print(f"[DEBUG] 开始扫描目录: {base_path}")
    results = defaultdict(lambda: defaultdict(dict))
    base = Path(base_path)
    if not base.exists():
        print(f"[WARN] 目录不存在: {base_path}")
        return results

    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        print(f"[DEBUG] 发现模型目录: {model_dir.name}")
        for score_file in model_dir.glob('*.json.score'):
            print(f"  [DEBUG] 发现文件: {score_file.name}")
            filename = score_file.name
            task_name = extract_task_name(filename)
            seqlen = extract_seqlen(filename)

            score_data = load_score_file(score_file)
            metric_value = get_metric_value(score_data, task_name)

            if metric_value is not None:
                results[model_dir.name][seqlen][task_name] = {
                    'score': metric_value,
                    'metric': DATASET_TO_METRICS.get(task_name),
                    'filename': filename
                }
            else:
                print(f"  [WARN] 未能获取指标值: {filename}")
    print(f"[DEBUG] 扫描完成，总模型数: {len(results)}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 模型长文本能力评估可视化工具")
    print("=" * 60)
    print("📁 请确保 'output' 目录存在并包含评估数据")
    print("🌐 访问地址: http://localhost:5000")
    print("=" * 60)

    if not os.path.exists('output'):
        print("⚠️  警告: 'output' 目录不存在，将创建空目录")
        os.makedirs('output')

    app.run(debug=True, host='0.0.0.0', port=8912)
