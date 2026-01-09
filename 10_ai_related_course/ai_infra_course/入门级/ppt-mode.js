/**
 * PPT模式控制器
 * 为Linux基础课程HTML页面提供PPT演示功能
 * 支持键盘导航、全屏模式、页面预览等功能
 */

class PPTController {
    constructor() {
        this.totalPages = 35;
        this.currentPage = 1;
        this.isFullscreen = false;
        this.isPPTMode = false;
        this.isNavigating = false;
        this.basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));

        this.init();
    }

    // 保存PPT模式状态到localStorage
    saveState() {
        console.log('=== 保存PPT状态 ===');
        const state = {
            isPPTMode: this.isPPTMode,
            currentPage: this.currentPage,
            timestamp: Date.now()
        };
        localStorage.setItem('ppt-mode-state', JSON.stringify(state));
        console.log('状态已保存:', state);
    }

    // 从localStorage加载PPT模式状态
    loadState() {
        console.log('=== 加载PPT状态 ===');
        try {
            const stateStr = localStorage.getItem('ppt-mode-state');
            if (stateStr) {
                const state = JSON.parse(stateStr);
                console.log('找到保存的状态:', state);

                // 检查状态是否过期（5分钟内有效）
                const now = Date.now();
                if (now - state.timestamp < 5 * 60 * 1000) {
                    this.isPPTMode = state.isPPTMode;
                    console.log('状态有效，恢复PPT模式:', this.isPPTMode);
                } else {
                    console.log('状态已过期，清除保存的状态');
                    localStorage.removeItem('ppt-mode-state');
                }
            } else {
                console.log('未找到保存的状态');
            }
        } catch (error) {
            console.log('加载状态时出错:', error);
            localStorage.removeItem('ppt-mode-state');
        }
    }

    init() {
        console.log('=== PPT控制器初始化开始 ===');
        console.log('当前页面URL:', window.location.href);
        console.log('基础路径:', this.basePath);
        console.log('当前页面编号:', this.currentPage);
        console.log('总页数:', this.totalPages);

        // 先加载状态
        this.loadState();

        this.createPPTInterface();
        console.log('PPT界面创建完成');

        this.bindEvents();
        console.log('事件绑定完成');

        this.detectCurrentPage();
        console.log('页面检测完成');

        // 如果之前是PPT模式，恢复状态
        if (this.isPPTMode) {
            console.log('恢复PPT模式状态');
            this.enterPPTMode();
        }

        console.log('=== PPT控制器初始化完成 ===');
    }

    // 检测当前页面编号
    detectCurrentPage() {
        const currentPath = window.location.pathname;
        const match = currentPath.match(/page_(\d+)\.html/);
        if (match) {
            this.currentPage = parseInt(match[1]);
        }
        this.updatePageIndicator();
    }

    // 创建PPT控制界面
    createPPTInterface() {
        // 创建PPT控制面板
        const controlPanel = document.createElement('div');
        controlPanel.id = 'ppt-control-panel';
        controlPanel.innerHTML = `
            <div class="ppt-controls">
                <button id="ppt-toggle" class="ppt-btn" title="切换PPT模式 (F5)">
                    <i class="fas fa-play"></i>
                </button>
                <button id="ppt-prev" class="ppt-btn" title="上一页 (←)">
                    <i class="fas fa-chevron-left"></i>
                </button>
                <span id="ppt-page-indicator" class="ppt-indicator">
                    ${this.currentPage} / ${this.totalPages}
                </span>
                <button id="ppt-next" class="ppt-btn" title="下一页 (→)">
                    <i class="fas fa-chevron-right"></i>
                </button>
                <button id="ppt-fullscreen" class="ppt-btn" title="全屏 (F11)">
                    <i class="fas fa-expand"></i>
                </button>
                <button id="ppt-overview" class="ppt-btn" title="页面概览 (O)">
                    <i class="fas fa-th"></i>
                </button>
            </div>
        `;

        // 添加样式
        const style = document.createElement('style');
        style.textContent = `
            #ppt-control-panel {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 10000;
                background: rgba(15, 23, 42, 0.95);
                border: 1px solid #4b5563;
                border-radius: 12px;
                padding: 10px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            #ppt-control-panel:hover {
                background: rgba(15, 23, 42, 1);
                transform: translateY(-2px);
            }

            .ppt-controls {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .ppt-btn {
                background: #374151;
                border: none;
                color: #e5e7eb;
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 14px;
            }

            .ppt-btn:hover {
                background: #3b82f6;
                transform: scale(1.05);
            }

            .ppt-btn:active {
                transform: scale(0.95);
            }

            .ppt-indicator {
                color: #e5e7eb;
                font-size: 14px;
                font-weight: 500;
                padding: 0 8px;
                min-width: 60px;
                text-align: center;
            }

            .ppt-mode-active {
                cursor: none;
            }

            .ppt-mode-active #ppt-control-panel {
                opacity: 0;
                pointer-events: none;
            }

            .ppt-mode-active #ppt-control-panel:hover {
                opacity: 1;
                pointer-events: all;
            }

            /* 移除页面切换动画以避免闪烁 */

            /* 概览模式样式 */
            #ppt-overview-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9);
                z-index: 20000;
                display: none;
                overflow-y: auto;
                padding: 20px;
            }

            .overview-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }

            .overview-page {
                background: #1e293b;
                border: 2px solid #374151;
                border-radius: 8px;
                padding: 10px;
                cursor: pointer;
                transition: all 0.2s ease;
                text-align: center;
            }

            .overview-page:hover {
                border-color: #3b82f6;
                transform: scale(1.05);
            }

            .overview-page.current {
                border-color: #f97316;
                background: #292524;
            }

            .overview-page-number {
                font-size: 18px;
                font-weight: bold;
                color: #3b82f6;
                margin-bottom: 5px;
            }

            .overview-page-title {
                font-size: 12px;
                color: #e5e7eb;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            /* 帮助提示 */
            .ppt-help {
                position: fixed;
                top: 20px;
                left: 20px;
                background: rgba(15, 23, 42, 0.95);
                color: #e5e7eb;
                padding: 15px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 15000;
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }

            .ppt-help.show {
                opacity: 1;
            }

            .ppt-help h4 {
                margin: 0 0 10px 0;
                color: #3b82f6;
            }

            .ppt-help ul {
                margin: 0;
                padding-left: 15px;
            }

            .ppt-help li {
                margin-bottom: 3px;
            }
        `;

        document.head.appendChild(style);
        document.body.appendChild(controlPanel);

        // 创建概览模态框
        this.createOverviewModal();

        // 创建帮助提示
        this.createHelpTooltip();
    }

    // 创建概览模态框
    createOverviewModal() {
        const modal = document.createElement('div');
        modal.id = 'ppt-overview-modal';
        modal.innerHTML = `
            <div class="overview-grid" id="overview-grid">
                <!-- 页面缩略图将在这里动态生成 -->
            </div>
        `;
        document.body.appendChild(modal);
    }

    // 创建帮助提示
    createHelpTooltip() {
        const help = document.createElement('div');
        help.className = 'ppt-help';
        help.id = 'ppt-help';
        help.innerHTML = `
            <h4>PPT模式快捷键</h4>
            <ul>
                <li>F5 - 切换PPT模式</li>
                <li>← / → - 上一页/下一页</li>
                <li>F11 - 全屏切换</li>
                <li>O - 页面概览</li>
                <li>H - 显示/隐藏帮助</li>
                <li>ESC - 退出模式</li>
            </ul>
        `;
        document.body.appendChild(help);
    }

    // 绑定事件
    bindEvents() {
        // 按钮事件
        document.getElementById('ppt-toggle').addEventListener('click', () => this.togglePPTMode());
        document.getElementById('ppt-prev').addEventListener('click', () => this.previousPage());
        document.getElementById('ppt-next').addEventListener('click', () => this.nextPage());
        document.getElementById('ppt-fullscreen').addEventListener('click', () => this.toggleFullscreen());
        document.getElementById('ppt-overview').addEventListener('click', () => this.showOverview());

        // 键盘事件 - 使用捕获阶段确保优先处理
        document.addEventListener('keydown', (e) => this.handleKeydown(e), true);
        document.addEventListener('keyup', (e) => this.handleKeyup(e), true);

        // 概览模态框点击事件
        document.getElementById('ppt-overview-modal').addEventListener('click', (e) => {
            if (e.target.id === 'ppt-overview-modal') {
                this.hideOverview();
            }
        });

        // 鼠标滚轮事件（在PPT模式下）
        document.addEventListener('wheel', (e) => {
            if (this.isPPTMode) {
                e.preventDefault();
                e.stopPropagation();
                if (e.deltaY > 0) {
                    this.nextPage();
                } else {
                    this.previousPage();
                }
            }
        }, { passive: false, capture: true });

        // 阻止PPT模式下的默认滚动行为
        document.addEventListener('scroll', (e) => {
            if (this.isPPTMode) {
                e.preventDefault();
                window.scrollTo(0, 0);
            }
        }, { passive: false });
    }

    // 键盘事件处理
    handleKeydown(e) {
        console.log('=== 键盘事件调试信息 ===');
        console.log('按键:', e.key);
        console.log('键码:', e.keyCode);
        console.log('事件类型:', e.type);
        console.log('PPT模式:', this.isPPTMode);
        console.log('事件目标:', e.target.tagName, e.target.className);
        console.log('默认行为已阻止:', e.defaultPrevented);
        console.log('事件传播已停止:', e.cancelBubble);
        console.log('当前页面:', this.currentPage);
        console.log('总页数:', this.totalPages);
        console.log('========================');

        // 在PPT模式下处理键盘事件
        if (this.isPPTMode) {
            console.log('进入PPT模式键盘处理逻辑');

            // 阻止所有可能的默认行为和事件传播
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            console.log('事件阻止后状态 - 默认行为已阻止:', e.defaultPrevented);

            switch (e.key) {
                case 'F5':
                    console.log('处理F5键 - 切换PPT模式');
                    this.togglePPTMode();
                    break;
                case 'ArrowLeft':
                case 'ArrowUp':
                case 'PageUp':
                    console.log('处理向前导航键:', e.key);
                    this.previousPage();
                    break;
                case 'ArrowRight':
                case 'ArrowDown':
                case 'PageDown':
                case ' ': // 空格键
                    console.log('处理向后导航键:', e.key);
                    this.nextPage();
                    break;
                case 'F11':
                    console.log('处理F11键 - 切换全屏');
                    this.toggleFullscreen();
                    break;
                case 'o':
                case 'O':
                    console.log('处理O键 - 显示概览');
                    this.showOverview();
                    break;
                case 'h':
                case 'H':
                    console.log('处理H键 - 切换帮助');
                    this.toggleHelp();
                    break;
                case 'Escape':
                    console.log('处理Escape键');
                    if (document.getElementById('ppt-overview-modal').style.display === 'block') {
                        console.log('隐藏概览模式');
                        this.hideOverview();
                    } else {
                        console.log('退出PPT模式');
                        this.togglePPTMode();
                    }
                    break;
                case 'Home':
                    console.log('处理Home键 - 跳转到第一页');
                    this.goToPage(1);
                    break;
                case 'End':
                    console.log('处理End键 - 跳转到最后一页');
                    this.goToPage(this.totalPages);
                    break;
                default:
                    console.log('未处理的按键:', e.key);
            }
        } else {
            console.log('非PPT模式，只处理F5键');
            // 非PPT模式下只处理F5键
            if (e.key === 'F5') {
                console.log('处理F5键 - 进入PPT模式');
                e.preventDefault();
                this.togglePPTMode();
            }
        }
    }

    // 键盘释放事件处理
    handleKeyup(e) {
        // 在PPT模式下阻止所有导航键的默认行为
        if (this.isPPTMode) {
            switch (e.key) {
                case 'ArrowLeft':
                case 'ArrowRight':
                case 'ArrowUp':
                case 'ArrowDown':
                case ' ':
                case 'PageUp':
                case 'PageDown':
                case 'Home':
                case 'End':
                    e.preventDefault();
                    e.stopPropagation();
                    break;
            }
        }
    }

    // 进入PPT模式
    enterPPTMode() {
        console.log('=== 进入PPT模式 ===');
        this.isPPTMode = true;

        const toggleBtn = document.getElementById('ppt-toggle');
        document.body.classList.add('ppt-mode-active');
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-stop"></i>';
            toggleBtn.title = '退出PPT模式 (F5)';
        }

        // 禁用页面滚动
        document.body.style.overflow = 'hidden';
        document.documentElement.style.overflow = 'hidden';
        console.log('页面滚动已禁用');

        // 滚动到页面顶部
        window.scrollTo(0, 0);
        console.log('滚动到页面顶部');

        this.showHelp();
        console.log('帮助信息已显示');

        this.saveState();
    }

    // 退出PPT模式
    exitPPTMode() {
        console.log('=== 退出PPT模式 ===');
        this.isPPTMode = false;

        const toggleBtn = document.getElementById('ppt-toggle');
        document.body.classList.remove('ppt-mode-active');
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-play"></i>';
            toggleBtn.title = '开始PPT模式 (F5)';
        }

        // 恢复页面滚动
        document.body.style.overflow = '';
        document.documentElement.style.overflow = '';
        console.log('页面滚动已恢复');

        this.hideHelp();
        console.log('帮助信息已隐藏');

        // 清除保存的状态
        localStorage.removeItem('ppt-mode-state');
        console.log('PPT状态已清除');
    }

    // 切换PPT模式
    togglePPTMode() {
        console.log('=== togglePPTMode 方法调用 ===');
        console.log('当前PPT模式状态:', this.isPPTMode);

        if (this.isPPTMode) {
            this.exitPPTMode();
        } else {
            this.enterPPTMode();
        }
    }

    // 上一页
    previousPage() {
        console.log('=== previousPage 方法调用 ===');
        console.log('当前页面:', this.currentPage);
        console.log('总页数:', this.totalPages);

        if (this.currentPage > 1) {
            console.log('可以向前翻页，跳转到页面:', this.currentPage - 1);
            this.goToPage(this.currentPage - 1);
        } else {
            console.log('已经是第一页，无法向前翻页');
        }
    }

    // 下一页
    nextPage() {
        console.log('=== nextPage 方法调用 ===');
        console.log('当前页面:', this.currentPage);
        console.log('总页数:', this.totalPages);

        if (this.currentPage < this.totalPages) {
            console.log('可以向后翻页，跳转到页面:', this.currentPage + 1);
            this.goToPage(this.currentPage + 1);
        } else {
            console.log('已经是最后一页，无法向后翻页');
        }
    }

    // 跳转到指定页面
    goToPage(pageNumber) {
        console.log('=== goToPage 方法调用 ===');
        console.log('目标页面:', pageNumber);
        console.log('当前页面:', this.currentPage);
        console.log('总页数:', this.totalPages);
        console.log('页面范围检查:', pageNumber >= 1 && pageNumber <= this.totalPages);

        if (pageNumber < 1 || pageNumber > this.totalPages) {
            console.log('页面范围无效，跳转失败');
            return;
        }

        console.log('页面范围有效，执行跳转');

        // 在跳转前保存当前状态（包括目标页面）
        if (this.isPPTMode) {
            console.log('PPT模式下跳转，保存状态');
            const state = {
                isPPTMode: true,
                currentPage: pageNumber, // 保存目标页面
                timestamp: Date.now()
            };
            localStorage.setItem('ppt-mode-state', JSON.stringify(state));
            console.log('跳转前状态已保存:', state);
        }

        const targetUrl = `${this.basePath}/page_${pageNumber}.html`;
        console.log('目标URL:', targetUrl);

        // 防止重复跳转
        if (this.isNavigating) {
            console.log('正在导航中，忽略重复请求');
            return;
        }
        this.isNavigating = true;

        // 直接跳转，不使用动画避免闪烁
        console.log('执行页面跳转到:', targetUrl);
        window.location.href = targetUrl;
    }

    // 切换全屏
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log('无法进入全屏模式:', err);
            });
            document.getElementById('ppt-fullscreen').innerHTML = '<i class="fas fa-compress"></i>';
        } else {
            document.exitFullscreen();
            document.getElementById('ppt-fullscreen').innerHTML = '<i class="fas fa-expand"></i>';
        }
    }

    // 显示概览
    showOverview() {
        const modal = document.getElementById('ppt-overview-modal');
        const grid = document.getElementById('overview-grid');

        // 生成页面缩略图
        grid.innerHTML = '';
        for (let i = 1; i <= this.totalPages; i++) {
            const pageDiv = document.createElement('div');
            pageDiv.className = `overview-page ${i === this.currentPage ? 'current' : ''}`;
            pageDiv.innerHTML = `
                <div class="overview-page-number">第 ${i} 页</div>
                <div class="overview-page-title">页面 ${i}</div>
            `;
            pageDiv.addEventListener('click', () => {
                this.hideOverview();
                this.goToPage(i);
            });
            grid.appendChild(pageDiv);
        }

        modal.style.display = 'block';
    }

    // 隐藏概览
    hideOverview() {
        document.getElementById('ppt-overview-modal').style.display = 'none';
    }

    // 显示帮助
    showHelp() {
        document.getElementById('ppt-help').classList.add('show');
        setTimeout(() => {
            this.hideHelp();
        }, 3000);
    }

    // 隐藏帮助
    hideHelp() {
        document.getElementById('ppt-help').classList.remove('show');
    }

    // 切换帮助显示
    toggleHelp() {
        const help = document.getElementById('ppt-help');
        if (help.classList.contains('show')) {
            this.hideHelp();
        } else {
            this.showHelp();
        }
    }

    // 更新页面指示器
    updatePageIndicator() {
        const indicator = document.getElementById('ppt-page-indicator');
        if (indicator) {
            indicator.textContent = `${this.currentPage} / ${this.totalPages}`;
        }
    }
}

// 页面加载完成后初始化PPT控制器
document.addEventListener('DOMContentLoaded', () => {
    // 初始化PPT控制器
    window.pptController = new PPTController();

    console.log('PPT模式已启用！使用F5键开始演示。');
});