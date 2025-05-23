import os
import json
import shutil
from pathlib import Path

class FileManager:
    def __init__(self, output_folder="./output"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
    
    def create_project_folder(self, project_name):
        """프로젝트 폴더 생성"""
        project_path = os.path.join(self.output_folder, project_name)
        os.makedirs(project_path, exist_ok=True)
        return project_path
    
    def save_content(self, content, file_path):
        """콘텐츠를 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def save_json(self, data, file_path):
        """데이터를 JSON 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
    
    def create_react_app(self, project_folder):
        """React 앱 생성 및 필요한 파일 설정"""
        # 기본 구조 생성
        src_folder = os.path.join(project_folder, "src")
        components_folder = os.path.join(src_folder, "components")
        public_folder = os.path.join(project_folder, "public")
        
        os.makedirs(components_folder, exist_ok=True)
        os.makedirs(public_folder, exist_ok=True)
        
        # index.html 생성 (public 폴더에)
        index_html = """<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Travel Magazine created with CrewAI"
    />
    <title>Travel Magazine</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
"""
        self.save_content(index_html, os.path.join(public_folder, "index.html"))
        
        # index.js 생성
        index_js = """import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""
        self.save_content(index_js, os.path.join(src_folder, "index.js"))
        
        # index.css 생성
        index_css = """body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
"""
        self.save_content(index_css, os.path.join(src_folder, "index.css"))
        
        # App.css 생성
        # App.css 생성 부분을 다음과 같이 수정
        app_css = """.App {
          text-align: center;
          background-color: #f5f5f5;
          min-height: 100vh;
        }

        .magazine-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .magazine-header {
          background: white;
          padding: 40px 20px;
          margin-bottom: 30px;
          border-radius: 10px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .magazine-header h1 {
          font-size: 2.5em;
          margin: 0 0 10px 0;
          color: #2c3e50;
        }

        .magazine-header p {
          font-style: italic;
          font-size: 1.2em;
          color: #7f8c8d;
          margin: 0;
        }

        .magazine-footer {
          background: white;
          padding: 30px 20px;
          margin-top: 30px;
          border-radius: 10px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          color: #7f8c8d;
        }

        /* 반응형 디자인 */
        @media (max-width: 768px) {
          .magazine-container {
            padding: 10px;
          }
          
          .magazine-header {
            padding: 20px 15px;
          }
          
          .magazine-header h1 {
            font-size: 2em;
          }
        }

        /* 각 섹션 기본 스타일 */
        .magazine-container > div:not(.magazine-header):not(.magazine-footer) {
          margin-bottom: 40px;
        }

        /* 이미지 반응형 처리 */
        img {
          max-width: 100%;
          height: auto;
        }

        /* 텍스트 가독성 향상 */
        p {
          line-height: 1.6;
        }

        h1, h2, h3, h4, h5, h6 {
          line-height: 1.3;
        }
        """     

        self.save_content(app_css, os.path.join(src_folder, "App.css"))
        
        # package.json 생성
        package_json = """{
  "name": "travel-magazine",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.5",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "styled-components": "^5.3.6",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ],
    "rules": {
      "no-unused-vars": "warn",
      "jsx-a11y/img-redundant-alt": "off"
    }
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
"""
        self.save_content(package_json, os.path.join(project_folder, "package.json"))
        
        return src_folder, components_folder


