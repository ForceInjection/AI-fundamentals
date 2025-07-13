import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, AgentStatus, BeliefType, Desire

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    title: str
    content: str
    url: str
    relevance_score: float
    source_type: str  # web, academic, news, etc.
    timestamp: datetime
    
class ResearchAgent(BaseAgent):
    """研究智能体 - 专门负责信息收集和研究任务"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # 研究相关配置
        self.max_search_results = config.get("max_search_results", 10)
        self.search_timeout = config.get("search_timeout", 30)
        self.min_relevance_score = config.get("min_relevance_score", 0.3)
        
        # 搜索历史和缓存
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.cache_ttl = timedelta(hours=config.get("cache_ttl_hours", 24))
        
        # 专业领域
        self.expertise_domains = config.get("expertise_domains", [])
        
        # 初始化研究能力
        self._initialize_research_capabilities()
    
    def _initialize_research_capabilities(self):
        """初始化研究能力"""
        self.capabilities.extend([
            "web_search",
            "content_analysis",
            "source_validation",
            "trend_analysis",
            "competitive_research"
        ])
        
        # 加载研究工具
        self.tools.update({
            "web_search": self._web_search,
            "analyze_content": self._analyze_content,
            "validate_source": self._validate_source,
            "extract_insights": self._extract_insights
        })
        
        # 添加初始研究信念
        self.add_belief("research_quality_threshold", 0.7, BeliefType.FACT, 1.0, "config")
        self.add_belief("max_research_depth", 3, BeliefType.FACT, 1.0, "config")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取研究智能体的性能指标"""
        base_metrics = super().get_performance_metrics()
        
        cache_size = len(self.search_cache)
        cache_hits = self.performance_metrics.get("cache_hits", 0)
        total_searches = self.performance_metrics.get("total_searches", 0)
        cache_hit_rate = (cache_hits / total_searches) * 100 if total_searches > 0 else 0

        research_metrics = {
            "cache_size": cache_size,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "total_searches": total_searches,
            "avg_search_time": self.performance_metrics.get("avg_search_time", 0),
            "successful_searches": self.performance_metrics.get("successful_searches", 0),
            "failed_searches": self.performance_metrics.get("failed_searches", 0)
        }
        
        base_metrics.update(research_metrics)
        return base_metrics
    
    async def execute_research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行研究任务（兼容性方法）"""
        query = task.get("query", "")
        depth = task.get("depth", "standard")
        sources = task.get("sources", ["web"])
        
        # 调用标准的 execute 方法
        result = await self.execute({
            "query": query,
            "research_type": "general",
            "depth": depth
        })
        
        # 转换为兼容格式
        if result.status == AgentStatus.COMPLETED:
            return {
                "summary": result.data.get("summary", "Research completed successfully"),
                "confidence": result.data.get("confidence", 0.8),
                "sources": result.data.get("sources", []),
                "insights": result.data.get("insights", []),
                "execution_time": result.execution_time
            }
        else:
            return {
                "summary": f"Research failed: {result.error_message}",
                "confidence": 0.0,
                "sources": [],
                "insights": [],
                "execution_time": result.execution_time
            }
    
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """执行研究任务"""
        query = input_data.get("query", "")
        research_type = input_data.get("research_type", "general")
        depth = input_data.get("depth", "standard")
        
        if not query:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                execution_time=0,
                error_message="No research query provided"
            )
        
        try:
            # 执行研究流程
            research_results = await self._conduct_research(query, research_type, depth)
            
            # 分析和验证结果
            analyzed_results = await self._analyze_research_results(research_results)
            
            # 生成研究报告
            report = await self._generate_research_report(query, analyzed_results)
            
            # 计算置信度
            confidence_score = self._calculate_research_confidence(analyzed_results)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={
                    "query": query,
                    "research_type": research_type,
                    "raw_results": research_results,
                    "analyzed_results": analyzed_results,
                    "report": report,
                    "sources_count": len(research_results),
                    "confidence_score": confidence_score
                },
                execution_time=0,  # 将在基类中设置
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Research execution failed: {str(e)}")
            raise
    
    async def _conduct_research(self, query: str, research_type: str, depth: str) -> List[SearchResult]:
        """执行研究"""
        # 检查缓存
        cache_key = f"{query}_{research_type}_{depth}"
        if cache_key in self.search_cache:
            cached_results = self.search_cache[cache_key]
            if cached_results and (datetime.now() - cached_results[0].timestamp) < self.cache_ttl:
                self.logger.info(f"Using cached results for query: {query}")
                return cached_results
        
        # 根据研究类型选择搜索策略
        search_strategies = self._get_search_strategies(research_type, depth)
        
        all_results = []
        for strategy in search_strategies:
            try:
                results = await self._execute_search_strategy(query, strategy)
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(f"Search strategy {strategy['name']} failed: {str(e)}")
        
        # 去重和排序
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)
        
        # 限制结果数量
        final_results = sorted_results[:self.max_search_results]
        
        # 缓存结果
        self.search_cache[cache_key] = final_results
        
        return final_results
    
    def _get_search_strategies(self, research_type: str, depth: str) -> List[Dict[str, Any]]:
        """获取搜索策略"""
        strategies = []
        
        if research_type == "market_research":
            strategies.extend([
                {"name": "industry_reports", "sources": ["industry_databases"], "weight": 0.8},
                {"name": "competitor_analysis", "sources": ["company_websites", "news"], "weight": 0.7},
                {"name": "trend_analysis", "sources": ["social_media", "forums"], "weight": 0.6}
            ])
        elif research_type == "academic":
            strategies.extend([
                {"name": "scholarly_articles", "sources": ["academic_databases"], "weight": 0.9},
                {"name": "research_papers", "sources": ["arxiv", "pubmed"], "weight": 0.8},
                {"name": "expert_opinions", "sources": ["expert_blogs", "conferences"], "weight": 0.6}
            ])
        else:  # general research
            strategies.extend([
                {"name": "web_search", "sources": ["search_engines"], "weight": 0.7},
                {"name": "news_search", "sources": ["news_sites"], "weight": 0.6},
                {"name": "reference_materials", "sources": ["wikis", "encyclopedias"], "weight": 0.5}
            ])
        
        # 根据深度调整策略
        if depth == "deep":
            for strategy in strategies:
                strategy["max_results"] = 20
                strategy["follow_links"] = True
        elif depth == "surface":
            for strategy in strategies:
                strategy["max_results"] = 5
                strategy["follow_links"] = False
        else:  # standard
            for strategy in strategies:
                strategy["max_results"] = 10
                strategy["follow_links"] = False
        
        return strategies
    
    async def _execute_search_strategy(self, query: str, strategy: Dict[str, Any]) -> List[SearchResult]:
        """执行搜索策略"""
        results = []
        
        # 模拟不同的搜索源
        for source in strategy["sources"]:
            source_results = await self._search_source(query, source, strategy)
            results.extend(source_results)
        
        return results
    
    async def _search_source(self, query: str, source: str, strategy: Dict[str, Any]) -> List[SearchResult]:
        """搜索特定源"""
        # 这里是模拟实现，实际应该调用真实的搜索API
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 模拟搜索结果
        mock_results = [
            SearchResult(
                title=f"Research result {i+1} for '{query}' from {source}",
                content=f"This is mock content for {query} from {source}. " * 10,
                url=f"https://{source}.com/article/{i+1}",
                relevance_score=0.9 - (i * 0.1),
                source_type=source,
                timestamp=datetime.now()
            )
            for i in range(min(strategy["max_results"], 3))
        ]
        
        # 过滤低相关性结果
        filtered_results = [
            result for result in mock_results 
            if result.relevance_score >= self.min_relevance_score
        ]
        
        return filtered_results
    
    async def _web_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Web搜索工具"""
        # 实际实现应该调用搜索引擎API
        # 这里提供一个基本的HTTP搜索示例
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                # 这里应该是真实的搜索API调用
                # 例如：Google Custom Search API, Bing Search API等
                
                # 模拟搜索结果
                results = [
                    SearchResult(
                        title=f"Web result {i+1}: {query}",
                        content=f"Content about {query} from web search result {i+1}",
                        url=f"https://example{i+1}.com",
                        relevance_score=0.8 - (i * 0.05),
                        source_type="web",
                        timestamp=datetime.now()
                    )
                    for i in range(max_results)
                ]
                
                return results
                
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重搜索结果"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def _analyze_research_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """分析研究结果"""
        if not results:
            return {"summary": "No results to analyze", "insights": [], "quality_score": 0.0}
        
        # 内容分析
        content_analysis = await self._analyze_content_quality(results)
        
        # 来源分析
        source_analysis = self._analyze_sources(results)
        
        # 趋势分析
        trend_analysis = self._analyze_trends(results)
        
        # 提取关键洞察
        insights = await self._extract_insights(results)
        
        # 计算整体质量分数
        quality_score = self._calculate_quality_score(content_analysis, source_analysis)
        
        return {
            "summary": self._generate_summary(results),
            "content_analysis": content_analysis,
            "source_analysis": source_analysis,
            "trend_analysis": trend_analysis,
            "insights": insights,
            "quality_score": quality_score,
            "total_results": len(results)
        }
    
    async def _analyze_content_quality(self, results: List[SearchResult]) -> Dict[str, Any]:
        """分析内容质量"""
        total_length = sum(len(result.content) for result in results)
        avg_length = total_length / len(results) if results else 0
        
        # 分析内容深度
        depth_scores = []
        for result in results:
            # 简单的深度评估：基于内容长度和关键词密度
            depth_score = min(len(result.content) / 1000, 1.0)  # 标准化到0-1
            depth_scores.append(depth_score)
        
        avg_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0
        
        return {
            "average_content_length": avg_length,
            "average_depth_score": avg_depth,
            "content_diversity": len(set(result.source_type for result in results)),
            "relevance_distribution": {
                "high": len([r for r in results if r.relevance_score > 0.8]),
                "medium": len([r for r in results if 0.5 < r.relevance_score <= 0.8]),
                "low": len([r for r in results if r.relevance_score <= 0.5])
            }
        }
    
    def _analyze_sources(self, results: List[SearchResult]) -> Dict[str, Any]:
        """分析信息来源"""
        source_types = {}
        for result in results:
            source_types[result.source_type] = source_types.get(result.source_type, 0) + 1
        
        # 计算来源可信度
        credibility_scores = {
            "academic": 0.9,
            "news": 0.7,
            "web": 0.6,
            "social_media": 0.4,
            "forums": 0.3
        }
        
        weighted_credibility = sum(
            credibility_scores.get(source_type, 0.5) * count 
            for source_type, count in source_types.items()
        ) / len(results) if results else 0
        
        return {
            "source_distribution": source_types,
            "source_diversity": len(source_types),
            "weighted_credibility": weighted_credibility,
            "total_sources": len(results)
        }
    
    def _analyze_trends(self, results: List[SearchResult]) -> Dict[str, Any]:
        """分析趋势"""
        # 时间分布分析
        time_distribution = {}
        now = datetime.now()
        
        for result in results:
            days_ago = (now - result.timestamp).days
            if days_ago <= 1:
                period = "today"
            elif days_ago <= 7:
                period = "this_week"
            elif days_ago <= 30:
                period = "this_month"
            else:
                period = "older"
            
            time_distribution[period] = time_distribution.get(period, 0) + 1
        
        return {
            "time_distribution": time_distribution,
            "freshness_score": time_distribution.get("today", 0) / len(results) if results else 0
        }
    
    async def _extract_insights(self, results: List[SearchResult]) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        # 简单的关键词提取和模式识别
        all_content = " ".join(result.content for result in results)
        
        # 提取高频词汇（简化实现）
        words = re.findall(r'\b\w+\b', all_content.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 过滤短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 获取最频繁的词汇
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_words:
            insights.append(f"Most frequently mentioned terms: {', '.join([word for word, _ in top_words[:5]])}")
        
        # 分析来源一致性
        if len(set(result.source_type for result in results)) > 3:
            insights.append("Information comes from diverse sources, indicating broad coverage")
        
        # 分析相关性分布
        high_relevance_count = len([r for r in results if r.relevance_score > 0.8])
        if high_relevance_count > len(results) * 0.7:
            insights.append("High proportion of highly relevant results suggests focused topic")
        
        return insights
    
    def _calculate_quality_score(self, content_analysis: Dict[str, Any], source_analysis: Dict[str, Any]) -> float:
        """计算研究质量分数"""
        # 综合评估各个维度
        content_score = content_analysis.get("average_depth_score", 0) * 0.3
        source_score = source_analysis.get("weighted_credibility", 0) * 0.4
        diversity_score = min(source_analysis.get("source_diversity", 0) / 5, 1.0) * 0.3
        
        return content_score + source_score + diversity_score
    
    def _calculate_research_confidence(self, analyzed_results: Dict[str, Any]) -> float:
        """计算研究置信度"""
        quality_score = analyzed_results.get("quality_score", 0)
        result_count = analyzed_results.get("total_results", 0)
        
        # 基于质量和数量计算置信度
        count_factor = min(result_count / 10, 1.0)  # 10个结果为满分
        confidence = (quality_score * 0.7) + (count_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def _generate_summary(self, results: List[SearchResult]) -> str:
        """生成研究摘要"""
        if not results:
            return "No research results available."
        
        summary_parts = [
            f"Research completed with {len(results)} sources.",
            f"Average relevance score: {sum(r.relevance_score for r in results) / len(results):.2f}",
            f"Source types: {', '.join(set(r.source_type for r in results))}"
        ]
        
        return " ".join(summary_parts)
    
    async def _generate_research_report(self, query: str, analyzed_results: Dict[str, Any]) -> str:
        """生成研究报告"""
        report_sections = [
            f"# Research Report: {query}",
            f"\n## Executive Summary",
            analyzed_results.get("summary", "No summary available."),
            f"\n## Key Insights",
        ]
        
        insights = analyzed_results.get("insights", [])
        if insights:
            for i, insight in enumerate(insights, 1):
                report_sections.append(f"{i}. {insight}")
        else:
            report_sections.append("No specific insights identified.")
        
        report_sections.extend([
            f"\n## Quality Assessment",
            f"- Overall Quality Score: {analyzed_results.get('quality_score', 0):.2f}/1.0",
            f"- Source Credibility: {analyzed_results.get('source_analysis', {}).get('weighted_credibility', 0):.2f}/1.0",
            f"- Content Depth: {analyzed_results.get('content_analysis', {}).get('average_depth_score', 0):.2f}/1.0",
            f"\n## Methodology",
            f"- Total sources analyzed: {analyzed_results.get('total_results', 0)}",
            f"- Source diversity: {analyzed_results.get('source_analysis', {}).get('source_diversity', 0)} different types",
            f"- Research confidence: {self._calculate_research_confidence(analyzed_results):.2f}/1.0"
        ])
        
        return "\n".join(report_sections)
    
    async def _analyze_content(self, content: str, **kwargs) -> Dict[str, Any]:
        """分析内容质量和相关性"""
        try:
            # 基本内容分析
            word_count = len(content.split())
            sentence_count = len([s for s in content.split('.') if s.strip()])
            
            # 计算内容深度分数
            depth_score = min(1.0, word_count / 500)  # 500词为满分
            
            # 计算可读性分数（简化版）
            avg_sentence_length = word_count / max(sentence_count, 1)
            readability_score = max(0, 1.0 - (avg_sentence_length - 15) / 20)
            
            # 检查关键词密度
            query = kwargs.get('query', '')
            keyword_density = 0
            if query:
                keywords = query.lower().split()
                content_lower = content.lower()
                keyword_matches = sum(content_lower.count(kw) for kw in keywords)
                keyword_density = min(1.0, keyword_matches / max(word_count / 100, 1))
            
            # 综合质量分数
            quality_score = (depth_score * 0.4 + readability_score * 0.3 + keyword_density * 0.3)
            
            return {
                "quality_score": quality_score,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "depth_score": depth_score,
                "readability_score": readability_score,
                "keyword_density": keyword_density,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {
                "quality_score": 0.0,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
     
    async def _validate_source(self, source_url: str, **kwargs) -> Dict[str, Any]:
        """验证信息源的可信度"""
        try:
            # 基本URL验证
            if not source_url or not source_url.startswith(('http://', 'https://')):
                return {
                    "credibility_score": 0.0,
                    "validation_status": "invalid_url",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 域名信任度评估（简化版）
            domain_trust_scores = {
                'edu': 0.9,
                'gov': 0.95,
                'org': 0.7,
                'com': 0.6,
                'net': 0.5
            }
            
            # 提取域名后缀
            domain_suffix = 'com'  # 默认值
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(source_url)
                domain_parts = parsed_url.netloc.split('.')
                if len(domain_parts) > 1:
                    domain_suffix = domain_parts[-1]
            except:
                pass
            
            credibility_score = domain_trust_scores.get(domain_suffix, 0.5)
            
            # 检查是否为知名可信网站
            trusted_domains = [
                'wikipedia.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov',
                'arxiv.org', 'nature.com', 'science.org', 'ieee.org'
            ]
            
            for trusted_domain in trusted_domains:
                if trusted_domain in source_url:
                    credibility_score = max(credibility_score, 0.9)
                    break
            
            return {
                "credibility_score": credibility_score,
                "domain_suffix": domain_suffix,
                "validation_status": "validated",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Source validation failed: {e}")
            return {
                "credibility_score": 0.0,
                "validation_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
     
    def _generate_goals(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成研究相关的目标"""
        goals = []
        
        # 如果信念中有待研究的主题
        research_topics = [belief.value for belief in self.beliefs.values() 
                         if belief.key.startswith("research_topic_")]
        
        for topic in research_topics:
            goals.append({
                "goal_id": f"research_{topic.replace(' ', '_')}",
                "description": f"Conduct comprehensive research on {topic}",
                "priority": 7,
                "success_criteria": {"min_sources": 5, "min_quality_score": 0.6}
            })
        
        return goals
    
    def _create_plan(self, goal: Desire) -> List[Dict[str, Any]]:
        """为研究目标创建执行计划"""
        if goal.goal_id.startswith("research_"):
            return [
                {"action": "web_search", "parameters": {"query": goal.description, "max_results": 10}},
                {"action": "analyze_content", "parameters": {"depth": "standard"}},
                {"action": "validate_source", "parameters": {"min_credibility": 0.5}},
                {"action": "extract_insights", "parameters": {"min_insights": 3}}
            ]
        
        return []