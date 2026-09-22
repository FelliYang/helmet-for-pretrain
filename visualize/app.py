#!/usr/bin/env python3
"""
模型评估结果可视化 Web 服务
使用方法: python app.py --dir /path/to/results --port 8080
"""

import os
import json
import argparse
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from typing import Dict, List, Any

app = Flask(__name__)

# 全局配置
CONFIG = {
    'data_dir': None,
    'parsed_data': None
}

# Benchmark 适配器配置
BENCHMARK_ADAPTERS = {
    'json_kv': {
        'name': 'JSON KV',
        'metrics_path': ['averaged_metrics'],
        'metrics': ['exact_match', 'f1', 'substring_exact_match', 'rougeL_f1', 'input_len', 'output_len'],
        'case_fields': [
            {'key': 'question', 'label': '问题', 'type': 'text'},
            {'key': 'answer', 'label': '正确答案', 'type': 'text'},
            {'key': 'parsed_output', 'label': '模型输出', 'type': 'text'},
            {'key': 'exact_match', 'label': 'Exact Match', 'type': 'metric'},
            {'key': 'f1', 'label': 'F1 Score', 'type': 'metric'},
            {'key': 'depth', 'label': 'Depth', 'type': 'number'},
            {'key': 'num_kvs', 'label': 'Num KVs', 'type': 'number'},
        ]
    },
    'ruler_mk_2': {
        'name': 'RULER MK-2',
        'metrics_path': ['averaged_metrics'],
        'metrics': ['ruler_recall', 'input_len', 'output_len'],
        'case_fields': [
            {'key': 'question', 'label': '问题', 'type': 'text'},
            {'key': 'answer', 'label': '正确答案', 'type': 'text'},
            {'key': 'parsed_output', 'label': '模型输出', 'type': 'text'},
            {'key': 'ruler_recall', 'label': 'Recall', 'type': 'metric'},
            {'key': 'type_needle_v', 'label': '类型', 'type': 'text'},
            {'key': 'length', 'label': 'Length', 'type': 'number'},
        ]
    }
}


def infer_benchmark_type(filename: str) -> str:
    """从文件名推断 benchmark 类型"""
    lower = filename.lower()
    for key in BENCHMARK_ADAPTERS.keys():
        if key in lower:
            return key
    return None


def load_directory(data_dir: Path) -> Dict[str, Dict]:
    """加载目录下的所有评估结果"""
    results = {}
    
    if not data_dir.exists():
        raise ValueError(f"目录不存在: {data_dir}")
    
    if not data_dir.is_dir():
        raise ValueError(f"路径不是目录: {data_dir}")
    
    # 遍历所有子目录
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        results[model_name] = {}
        
        # 查找所有 JSON 文件
        for json_file in model_dir.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                benchmark_type = infer_benchmark_type(json_file.name)
                if not benchmark_type:
                    print(f"⚠️  跳过未识别的文件: {json_file.name}")
                    continue
                
                adapter = BENCHMARK_ADAPTERS[benchmark_type]
                
                # 提取指标
                metrics = {}
                avg_metrics = data.get('averaged_metrics', {})
                for metric in adapter['metrics']:
                    metrics[metric] = avg_metrics.get(metric)
                
                # 提取 cases
                cases = data.get('data', [])
                
                results[model_name][benchmark_type] = {
                    'metrics': metrics,
                    'cases': cases,
                    'raw': data,
                    'adapter': adapter
                }
                
                print(f"✓ 加载: {model_name}/{json_file.name} ({len(cases)} cases)")
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON 解析失败: {json_file.name} - {e}")
            except Exception as e:
                print(f"✗ 读取失败: {json_file.name} - {e}")
    
    if not results:
        raise ValueError("未找到任何有效的评估结果文件")
    
    return results


def generate_summary_table(parsed_data: Dict) -> List[Dict]:
    """生成汇总表格数据"""
    rows = []
    
    for model_name, benchmarks in parsed_data.items():
        row = {'model': model_name}
        
        for bench_type, bench_data in benchmarks.items():
            for metric, value in bench_data['metrics'].items():
                key = f"{bench_type}_{metric}"
                row[key] = value
        
        rows.append(row)
    
    return rows


# ============================================================================
# HTML 模板
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型评估结果可视化</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-high { color: #16a34a; font-weight: 600; }
        .metric-low { color: #dc2626; }
        .sticky-col { position: sticky; left: 0; background: white; z-index: 10; }
        .sticky-header { position: sticky; top: 0; background: #f1f5f9; z-index: 20; }
    </style>
</head>
<body class="bg-slate-50">
    <div id="app"></div>
    
    <script>
        const PARSED_DATA = {{ parsed_data_json }};
        const SUMMARY_DATA = {{ summary_data_json }};
        const ADAPTERS = {{ adapters_json }};
    </script>
    
    <script>
        // 状态管理
        let state = {
            view: 'summary', // 'summary', 'model', 'cases'
            selectedModel: null,
            selectedBenchmark: null,
            searchTerm: '',
            sortKey: null,
            sortDir: 'asc',
            filterMetric: null
        };

        function setState(updates) {
            state = { ...state, ...updates };
            render();
        }

        // 排序功能
        function sortData(data, key, dir) {
            return [...data].sort((a, b) => {
                const aVal = a[key] ?? -Infinity;
                const bVal = b[key] ?? -Infinity;
                return dir === 'asc' ? aVal - bVal : bVal - aVal;
            });
        }

        // 格式化数字
        function formatValue(val) {
            if (val === null || val === undefined) return '-';
            if (typeof val === 'number') return val.toFixed(2);
            return val;
        }

        // 获取值的 CSS 类
        function getValueClass(val) {
            if (typeof val !== 'number') return '';
            if (val >= 80) return 'metric-high';
            if (val < 50) return 'metric-low';
            return '';
        }

        // 导出 CSV
        function exportCSV() {
            const headers = ['model', ...Object.keys(SUMMARY_DATA[0]).filter(k => k !== 'model')];
            const rows = SUMMARY_DATA.map(row => 
                headers.map(h => row[h] ?? '').join(',')
            );
            const csv = [headers.join(','), ...rows].join('\\n');
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'model_evaluation_summary.csv';
            a.click();
        }

        // 渲染汇总表格
        function renderSummary() {
            const columns = Object.keys(SUMMARY_DATA[0] || {}).filter(k => k !== 'model');
            const sortedData = state.sortKey ? sortData(SUMMARY_DATA, state.sortKey, state.sortDir) : SUMMARY_DATA;
            
            return `
                <div class="min-h-screen bg-slate-50 p-6">
                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-4">
                            <h1 class="text-3xl font-bold text-slate-800">评估结果总览</h1>
                            <button onclick="exportCSV()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                                导出 CSV
                            </button>
                        </div>
                        <p class="text-slate-600">已加载 ${Object.keys(PARSED_DATA).length} 个模型的评估结果</p>
                    </div>
                    
                    <div class="bg-white rounded-xl shadow-lg overflow-hidden">
                        <div class="overflow-x-auto">
                            <table class="w-full">
                                <thead class="bg-slate-100 border-b border-slate-200 sticky-header">
                                    <tr>
                                        <th class="px-4 py-3 text-left text-xs font-semibold text-slate-700 uppercase sticky-col">
                                            模型
                                        </th>
                                        ${columns.map(col => `
                                            <th onclick="setState({ sortKey: '${col}', sortDir: state.sortKey === '${col}' && state.sortDir === 'asc' ? 'desc' : 'asc' })" 
                                                class="px-4 py-3 text-left text-xs font-semibold text-slate-700 uppercase cursor-pointer hover:bg-slate-200">
                                                ${col.replace(/_/g, ' ')}
                                                ${state.sortKey === col ? (state.sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                                            </th>
                                        `).join('')}
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-200">
                                    ${sortedData.map(row => `
                                        <tr onclick="setState({ view: 'model', selectedModel: '${row.model}' })" 
                                            class="hover:bg-blue-50 cursor-pointer">
                                            <td class="px-4 py-3 text-sm font-medium text-slate-900 sticky-col">
                                                ${row.model}
                                            </td>
                                            ${columns.map(col => `
                                                <td class="px-4 py-3 text-sm text-slate-600">
                                                    <span class="${getValueClass(row[col])}">${formatValue(row[col])}</span>
                                                </td>
                                            `).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }

        // 渲染模型详情
        function renderModel() {
            const modelData = PARSED_DATA[state.selectedModel];
            
            return `
                <div class="min-h-screen bg-slate-50 p-6">
                    <div class="bg-white rounded-xl shadow-lg p-6">
                        <button onclick="setState({ view: 'summary', selectedModel: null })" 
                                class="mb-4 text-blue-600 hover:text-blue-800 font-medium">
                            ← 返回总览
                        </button>
                        
                        <h2 class="text-2xl font-bold text-slate-800 mb-6">${state.selectedModel}</h2>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            ${Object.entries(modelData).map(([bench, data]) => `
                                <div onclick="setState({ view: 'cases', selectedBenchmark: '${bench}' })"
                                     class="p-6 border-2 border-slate-200 rounded-lg hover:border-blue-400 hover:shadow-md cursor-pointer">
                                    <h3 class="text-lg font-semibold text-slate-800 mb-3">${data.adapter.name}</h3>
                                    <div class="space-y-2">
                                        ${Object.entries(data.metrics).map(([key, val]) => `
                                            <div class="flex justify-between text-sm">
                                                <span class="text-slate-600">${key}:</span>
                                                <span class="font-medium">${formatValue(val)}</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                    <div class="mt-4 text-xs text-slate-500">${data.cases.length} cases</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        }

        // 渲染 Cases
        function renderCases() {
            const benchData = PARSED_DATA[state.selectedModel][state.selectedBenchmark];
            const adapter = benchData.adapter;
            
            let filteredCases = benchData.cases;
            
            // 搜索过滤
            if (state.searchTerm) {
                const term = state.searchTerm.toLowerCase();
                filteredCases = filteredCases.filter(c => 
                    JSON.stringify(c).toLowerCase().includes(term)
                );
            }
            
            return `
                <div class="min-h-screen bg-slate-50 p-6">
                    <div class="bg-white rounded-xl shadow-lg p-6">
                        <button onclick="setState({ view: 'model', selectedBenchmark: null })" 
                                class="mb-4 text-blue-600 hover:text-blue-800 font-medium">
                            ← 返回 ${state.selectedModel}
                        </button>
                        
                        <h2 class="text-2xl font-bold text-slate-800 mb-2">
                            ${adapter.name} - ${state.selectedModel}
                        </h2>
                        <p class="text-slate-600 mb-6">共 ${filteredCases.length} 个 cases</p>
                        
                        <div class="mb-6">
                            <input type="text" 
                                   placeholder="搜索 cases..." 
                                   oninput="setState({ searchTerm: this.value })"
                                   value="${state.searchTerm}"
                                   class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        
                        <div class="space-y-4">
                            ${filteredCases.map((c, idx) => renderCase(c, idx, adapter)).join('')}
                        </div>
                    </div>
                </div>
            `;
        }

        // 渲染单个 Case
        function renderCase(caseData, index, adapter) {
            const caseId = `case-${index}`;
            
            return `
                <div class="border border-slate-200 rounded-lg overflow-hidden">
                    <div onclick="toggleCase('${caseId}')" 
                         class="flex justify-between items-center p-4 bg-slate-50 cursor-pointer hover:bg-slate-100">
                        <div class="flex items-center gap-3">
                            <span class="font-semibold text-slate-800">Case ${index + 1}</span>
                        </div>
                        <div class="flex gap-3">
                            ${adapter.case_fields.filter(f => f.type === 'metric').map(field => {
                                const val = caseData[field.key];
                                return `
                                    <div class="text-sm">
                                        <span class="text-slate-600">${field.label}: </span>
                                        <span class="${getValueClass(val * 100)}">
                                            ${val !== undefined ? val.toFixed(3) : '-'}
                                        </span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                    <div id="${caseId}" class="hidden p-4 space-y-4 bg-white">
                        ${adapter.case_fields.map(field => {
                            const val = caseData[field.key];
                            if (val === undefined || val === null) return '';
                            
                            return `
                                <div>
                                    <div class="text-xs font-semibold text-slate-500 uppercase mb-1">${field.label}</div>
                                    <div class="${field.type === 'text' ? 'p-3 bg-slate-50 rounded border border-slate-200 text-sm text-slate-700 whitespace-pre-wrap' : 'text-sm text-slate-800'}">
                                        ${typeof val === 'object' ? JSON.stringify(val, null, 2) : val}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        function toggleCase(id) {
            const el = document.getElementById(id);
            el.classList.toggle('hidden');
        }

        // 主渲染函数
        function render() {
            let content = '';
            
            if (state.view === 'summary') {
                content = renderSummary();
            } else if (state.view === 'model') {
                content = renderModel();
            } else if (state.view === 'cases') {
                content = renderCases();
            }
            
            document.getElementById('app').innerHTML = content;
        }

        // 初始化
        render();
    </script>
</body>
</html>
'''


# ============================================================================
# Flask 路由
# ============================================================================

@app.route('/')
def index():
    """主页面"""
    if CONFIG['parsed_data'] is None:
        return "<h1>错误：未指定数据目录</h1><p>请使用 --dir 参数指定目录</p>", 500
    
    summary_data = generate_summary_table(CONFIG['parsed_data'])
    
    # 手动序列化 JSON 以避免 Jinja2 自动转义问题
    import json
    parsed_data_json = json.dumps(CONFIG['parsed_data'], ensure_ascii=False)
    summary_data_json = json.dumps(summary_data, ensure_ascii=False)
    adapters_json = json.dumps(BENCHMARK_ADAPTERS, ensure_ascii=False)
    
    # 使用 Markup 标记为安全的 HTML
    from markupsafe import Markup
    
    return render_template_string(
        HTML_TEMPLATE,
        parsed_data_json=Markup(parsed_data_json),
        summary_data_json=Markup(summary_data_json),
        adapters_json=Markup(adapters_json)
    )


@app.route('/api/refresh', methods=['POST'])
def refresh():
    """重新加载数据"""
    try:
        CONFIG['parsed_data'] = load_directory(CONFIG['data_dir'])
        return jsonify({'status': 'ok', 'message': '数据已刷新'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='模型评估结果可视化服务')
    parser.add_argument('--dir', required=True, help='评估结果目录路径')
    parser.add_argument('--port', type=int, default=8080, help='服务端口（默认: 8080）')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址（默认: 0.0.0.0）')
    
    args = parser.parse_args()
    
    # 加载数据
    data_dir = Path(args.dir).expanduser().resolve()
    print(f"\n{'='*60}")
    print("🚀 模型评估结果可视化服务")
    print(f"{'='*60}")
    print(f"📁 数据目录: {data_dir}")
    print(f"\n正在加载数据...")
    
    try:
        CONFIG['data_dir'] = data_dir
        CONFIG['parsed_data'] = load_directory(data_dir)
        
        print(f"\n✓ 成功加载 {len(CONFIG['parsed_data'])} 个模型的数据")
        print(f"\n{'='*60}")
        print(f"✓ 服务运行在: http://{args.host}:{args.port}")
        print(f"{'='*60}")
        print("\n在浏览器中打开上述地址即可访问")
        print("按 Ctrl+C 停止服务\n")
        
        app.run(host=args.host, port=args.port, debug=False)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
