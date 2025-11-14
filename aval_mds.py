# aval-mds.py — v4.4 (critérios + rate-limit friendly + diag de auth + OpenAI integration + filtro por data de criação + explicação geral + suporte co-authors + alerta frequência commits)
# Requer: PyGithub, pandas, openpyxl, numpy, openai
# Uso:
#   python aval-mds.py --org unb-mds --since 2024-08-01 --out saida.xlsx --workers 8 --debug-auth --openai-key YOUR_KEY
#   python aval-mds.py --org unb-mds --since 2024-08-01 --include-new-repos --out saida.xlsx  # inclui repos criados desde a data
#   ou definir OPENAI_API_KEY como variável de ambiente

import argparse, datetime as dt, re, math, statistics, os, time, sys, json
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from github import Github, GithubException, UnknownObjectException

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[warning] OpenAI not available. Install with: pip install openai")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[warning] python-dotenv not available. Install with: pip install python-dotenv")

# ------------------ Config de critérios (igual às versões anteriores) ------------------

CRITERIA_WEIGHTS = {
    "weekly_commits": 0.08,
    "weekly_issues":  0.08,
    "weekly_prs":     0.08,
    "weekly_issue_comments": 0.04,
    "mature_pr_ratio":    0.12,
    "mature_issue_ratio": 0.12,
    "oss_signals": 0.12,
    "atomic_commits": 0.08,
    "issue_type_diversity": 0.04,
    "commit_curve_stability": 0.04,
    "ai_commit_quality": 0.08,
    "ai_issue_quality": 0.08,
    "ai_review_quality": 0.08,
    "rituals_participation": 0.00,
    "sprint_retro_regression": 0.00,
    "scrum_metrics_review": 0.00,
}

WEEKLY_TARGETS = {"commits": 1, "issues": 2, "prs": 1, "issue_comments": 1}
MIN_MATURE_COMMENTS = 2
ISSUE_ACCEPTANCE_HINTS = re.compile(r"\b(aceitaç|aceitacao|aceitação|crit[eé]rio[s]?|quality|qa|done|aceite)\b", re.I)
SAMPLE_COMMITS_PER_AUTHOR = 10
MAX_FILES_GOOD = 10
MIN_MSG_LEN = 8
STABILITY_MIN_WEEKS = 6

# ------------------ Rate limit helpers ------------------

def rate_limit_guard(gh: Github, threshold=100):
    try:
        core = gh.get_rate_limit().core
        if core.remaining < threshold:
            reset_in = (core.reset - dt.datetime.now(dt.timezone.utc)).total_seconds()
            wait = max(0, int(reset_in) + 2)
            print(f"[rate] remaining={core.remaining}. Aguardando reset em ~{wait}s...")
            time.sleep(wait)
    except Exception:
        pass

# ------------------ OpenAI Integration ------------------

class AgileQualityAnalyzer:
    """AI-powered analyzer for agile practice quality assessment."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.enabled = False
        
        if not OPENAI_AVAILABLE:
            return
            
        if api_key or os.getenv("OPENAI_API_KEY"):
            try:
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                # Test the connection
                self.enabled = True
                print("[ai] OpenAI client initialized successfully")
            except Exception as e:
                print(f"[ai] Failed to initialize OpenAI client: {e}")
    
    def _call_openai(self, messages: List[Dict], max_tokens: int = 150) -> Optional[str]:
        """Safe OpenAI API call with error handling."""
        if not self.enabled:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ai] OpenAI API error: {e}")
            return None
    
    def analyze_commit_quality(self, commit_messages: List[str]) -> float:
        """Analyze commit message quality using AI."""
        if not self.enabled or not commit_messages:
            return 0.5
        
        # Sample up to 5 recent commits for analysis
        sample = commit_messages[:5]
        messages_text = "\n".join([f"- {msg}" for msg in sample])
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in agile software development and commit message quality. "
                          "Evaluate commit messages based on: clarity, specificity, follows conventional commits, "
                          "describes 'what' and 'why', atomic changes. Return only a score from 0.0 to 1.0."
            },
            {
                "role": "user",
                "content": f"Rate these commit messages (0.0-1.0):\n{messages_text}"
            }
        ]
        
        result = self._call_openai(messages, max_tokens=50)
        if result:
            try:
                # Extract numeric score from response
                score_match = re.search(r'(\d+\.?\d*)', result)
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return 0.5
    
    def analyze_issue_quality(self, issue_texts: List[str]) -> float:
        """Analyze issue/story quality using AI."""
        if not self.enabled or not issue_texts:
            return 0.5
        
        # Sample up to 3 issues for analysis
        sample = issue_texts[:3]
        issues_text = "\n---\n".join(sample)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in agile user stories and issue quality. "
                          "Evaluate based on: clear acceptance criteria, user story format, "
                          "sufficient detail, testable requirements. Return only a score from 0.0 to 1.0."
            },
            {
                "role": "user",
                "content": f"Rate these issues/stories (0.0-1.0):\n{issues_text}"
            }
        ]
        
        result = self._call_openai(messages, max_tokens=50)
        if result:
            try:
                score_match = re.search(r'(\d+\.?\d*)', result)
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return 0.5
    
    def analyze_review_quality(self, review_comments: List[str]) -> float:
        """Analyze code review quality using AI."""
        if not self.enabled or not review_comments:
            return 0.5
        
        # Sample up to 5 review comments
        sample = review_comments[:5]
        comments_text = "\n".join([f"- {comment}" for comment in sample])
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in code review quality. "
                          "Evaluate based on: constructive feedback, specific suggestions, "
                          "addresses code quality/security/performance, collaborative tone. "
                          "Return only a score from 0.0 to 1.0."
            },
            {
                "role": "user",
                "content": f"Rate these code review comments (0.0-1.0):\n{comments_text}"
            }
        ]
        
        result = self._call_openai(messages, max_tokens=50)
        if result:
            try:
                score_match = re.search(r'(\d+\.?\d*)', result)
                if score_match:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return 0.5
    
    def generate_recommendations(self, user_metrics: Dict) -> str:
        """Generate personalized agile practice recommendations."""
        if not self.enabled:
            return "AI recommendations not available"
        
        metrics_summary = f"""
        Score: {user_metrics.get('score', 0):.2f}
        Commits: {user_metrics.get('commits', 0)}
        Issues: {user_metrics.get('issues', 0)}
        PRs: {user_metrics.get('prs', 0)}
        Commit Quality: {user_metrics.get('ai_commit_quality', 0.5):.2f}
        Issue Quality: {user_metrics.get('ai_issue_quality', 0.5):.2f}
        Review Quality: {user_metrics.get('ai_review_quality', 0.5):.2f}
        """
        
        messages = [
            {
                "role": "system",
                "content": "You are an agile coach. Provide 2-3 specific, actionable recommendations "
                          "to improve agile practices based on the metrics. Be concise and practical."
            },
            {
                "role": "user",
                "content": f"Suggest improvements for this developer:\n{metrics_summary}"
            }
        ]
        
        result = self._call_openai(messages, max_tokens=200)
        return result or "Focus on consistent contributions and quality documentation."

# ------------------ Repo signals ------------------

def has_file(repo, patterns):
    for p in patterns:
        try:
            repo.get_contents(p)
            return True
        except UnknownObjectException:
            continue
        except GithubException:
            continue
    return False

def compute_repo_signals(repo):
    readme = has_file(repo, ["README.md", "README.rst", "Readme.md", "readme.md"])
    license_ = has_file(repo, ["LICENSE", "LICENSE.md", "license", "LICENSE.txt"])
    contributing = has_file(repo, [".github/CONTRIBUTING.md", "CONTRIBUTING.md"])
    workflows = has_file(repo, [".github/workflows"])
    tests = has_file(repo, ["tests", "test", "spec"])
    coverage = has_file(repo, [".coveragerc", ".codecov.yml", ".nycrc", "coverage.xml"])
    dev_env = has_file(repo, ["docker-compose.yml", "Dockerfile", "Makefile", "devcontainer.json"])
    deploy_hints = has_file(repo, ["deploy", "charts", "helm", ".github/workflows/deploy.yml"])
    readme_run = False
    try:
        if readme:
            content = repo.get_readme().decoded_content.decode("utf-8", errors="ignore")
            readme_run = bool(re.search(r"(run|execute|como rodar|como executar|instala)", content, re.I))
    except Exception:
        pass

    return {
        "has_readme": readme, "has_license": license_, "has_contributing": contributing,
        "has_dev_env": dev_env, "has_tests": tests, "has_workflows": workflows,
        "has_coverage": coverage, "has_deploy_hints": deploy_hints,
        "has_readme_with_run": readme_run,
    }

def oss_signal_score(signals: dict):
    vals = list(signals.values())
    return sum(1 for v in vals if v) / max(1, len(vals))

# ------------------ Coleta de métricas ------------------

def week_index(d: dt.datetime, since_dt: dt.datetime):
    delta = d - since_dt
    return max(0, delta.days // 7)

def safe_login(u):
    return (getattr(u, "login", None) or "").lower() if u else None

def extract_co_authors_from_commit(commit_message: str) -> List[str]:
    """Extrai co-authors de uma mensagem de commit."""
    co_authors = []
    if not commit_message:
        return co_authors
    
    # Padrão: Co-authored-by: Name <email@domain.com>
    # ou Co-authored-by: username <email@domain.com>
    import re
    co_author_pattern = r'Co-authored-by:\s*([^<]+)\s*<([^>]+)>'
    matches = re.findall(co_author_pattern, commit_message, re.IGNORECASE)
    
    for name, email in matches:
        name = name.strip()
        email = email.strip().lower()
        
        # Tentar extrair username do email (comum em GitHub)
        if '@' in email:
            username_candidate = email.split('@')[0]
            # Se parece com um username GitHub (sem espaços, caracteres especiais básicos)
            if re.match(r'^[a-zA-Z0-9._-]+$', username_candidate):
                co_authors.append(username_candidate.lower())
        
        # Também tentar usar o nome se parece com username
        if re.match(r'^[a-zA-Z0-9._-]+$', name) and len(name) <= 39:  # GitHub username max length
            co_authors.append(name.lower())
    
    return list(set(co_authors))  # Remove duplicatas

def text_nontrivial(s: str):
    s = (s or "").strip()
    return len(s) >= 10

def collect_repo_contrib(repo, since_dt, ai_analyzer: Optional[AgileQualityAnalyzer] = None):
    signals = compute_repo_signals(repo)
    commits_by_user = defaultdict(int)
    issues_by_user  = defaultdict(int)
    prs_by_user     = defaultdict(int)
    mature_issue_by_user = defaultdict(int)
    mature_pr_by_user    = defaultdict(int)
    issue_comments_by_user_per_week = defaultdict(lambda: Counter())
    issue_type_diversity_by_user = defaultdict(lambda: Counter())
    atomicity_by_user = {}
    team_weekly_commits = Counter()

    # AI analysis data collection
    commit_messages_by_user = defaultdict(list)
    issue_texts_by_user = defaultdict(list)
    review_comments_by_user = defaultdict(list)

    author_to_recent_commits = defaultdict(list)
    processed_commits = 0

    try:
        for c in repo.get_commits(since=since_dt):
            if not c.commit or not c.commit.author or not c.commit.author.date:
                continue
            processed_commits += 1
            w = week_index(c.commit.author.date.replace(tzinfo=dt.timezone.utc), since_dt)
            team_weekly_commits[w] += 1
            # Processar autor principal
            author = safe_login(c.author)
            commit_msg = (c.commit.message or "").strip()
            
            # Lista de todos os autores (principal + co-authors)
            all_authors = []
            if author:
                all_authors.append(author)
            
            # Extrair co-authors da mensagem do commit
            co_authors = extract_co_authors_from_commit(commit_msg)
            all_authors.extend(co_authors)
            
            # Processar métricas para todos os autores (principal + co-authors)
            for current_author in all_authors:
                if not current_author:
                    continue
                    
                commits_by_user[current_author] += 1
                
                # Collect commit message for AI analysis
                if commit_msg and len(commit_messages_by_user[current_author]) < SAMPLE_COMMITS_PER_AUTHOR:
                    commit_messages_by_user[current_author].append(commit_msg)
                
                if len(author_to_recent_commits[current_author]) < SAMPLE_COMMITS_PER_AUTHOR:
                    author_to_recent_commits[current_author].append(c)
            if processed_commits % 50 == 0:
                time.sleep(0.2)
    except GithubException:
        pass

    for author, sample in author_to_recent_commits.items():
        if not sample:
            atomicity_by_user[author] = 0.5
            continue
        good = 0
        for c in sample:
            try:
                files = getattr(c, "files", None) or []
                # PaginatedList may not support len(); count by iteration safely
                try:
                    nfiles = sum(1 for _ in files)
                except TypeError:
                    nfiles = 0
                msg = (c.commit.message or "").strip()
                if 1 <= nfiles <= MAX_FILES_GOOD and len(msg) >= MIN_MSG_LEN:
                    good += 1
            except GithubException:
                continue
        atomicity_by_user[author] = good / max(1, len(sample))

    try:
        for idx, it in enumerate(repo.get_issues(state="all", since=since_dt)):
            author = safe_login(it.user)
            if not author:
                if idx % 30 == 0:
                    time.sleep(0.2)
                continue
            try:
                for cm in it.get_comments():
                    if cm.created_at and cm.user:
                        cm_author = safe_login(cm.user)
                        w = week_index(cm.created_at.replace(tzinfo=dt.timezone.utc), since_dt)
                        issue_comments_by_user_per_week[cm_author][w] += 1
                        
                        # Collect comment for AI analysis (if it's a review-like comment)
                        comment_body = (cm.body or "").strip()
                        if comment_body and len(comment_body) > 20:  # Filter short comments
                            if len(review_comments_by_user[cm_author]) < 5:
                                review_comments_by_user[cm_author].append(comment_body)
            except GithubException:
                pass

            if it.pull_request is None:
                issues_by_user[author] += 1
                
                # Collect issue text for AI analysis
                issue_body = (it.body or "").strip()
                if issue_body and len(issue_texts_by_user[author]) < 3:
                    issue_texts_by_user[author].append(f"{it.title}\n{issue_body}")
                try:
                    body_ok = text_nontrivial(it.body) and bool(ISSUE_ACCEPTANCE_HINTS.search(it.body or ""))
                    comments = list(it.get_comments())
                    commenters = {safe_login(c.user) for c in comments if c.user}
                    third_party = len([u for u in commenters if u and u != author])
                    if body_ok and third_party >= MIN_MATURE_COMMENTS:
                        mature_issue_by_user[author] += 1
                    for lb in it.labels or []:
                        name = (lb.name or "").lower()
                        if any(k in name for k in ["tech", "bug", "feature", "enhancement", "refactor"]):
                            issue_type_diversity_by_user[author]["tecnica"] += 1
                        elif "doc" in name or "docs" in name:
                            issue_type_diversity_by_user[author]["documentacao"] += 1
                        elif "devops" in name or "infra" in name or "ci" in name:
                            issue_type_diversity_by_user[author]["devops"] += 1
                        elif "manage" in name or "gest" in name or "product" in name or "prod" in name:
                            issue_type_diversity_by_user[author]["gestao"] += 1
                except GithubException:
                    pass
            else:
                prs_by_user[author] += 1
                
                # Collect PR text for AI analysis
                pr_body = (it.body or "").strip()
                if pr_body and len(issue_texts_by_user[author]) < 3:
                    issue_texts_by_user[author].append(f"{it.title}\n{pr_body}")
                
                try:
                    pr = repo.get_pull(it.number)
                    body_ok = text_nontrivial(pr.body)
                    comments = list(pr.get_comments()) + list(pr.get_review_comments())
                    
                    # Collect PR review comments for AI analysis
                    for comment in comments:
                        if comment.user:
                            comment_author = safe_login(comment.user)
                            comment_body = (comment.body or "").strip()
                            if comment_body and len(comment_body) > 20:
                                if len(review_comments_by_user[comment_author]) < 5:
                                    review_comments_by_user[comment_author].append(comment_body)
                    
                    commenters = {safe_login(c.user) for c in comments if c.user}
                    third_party = len([u for u in commenters if u and u != author])
                    if body_ok and third_party >= MIN_MATURE_COMMENTS:
                        mature_pr_by_user[author] += 1
                except GithubException:
                    pass
            if idx % 30 == 0:
                time.sleep(0.2)
    except GithubException:
        pass

    repo_meta = {
        "repo_full_name": repo.full_name,
        "default_branch": getattr(repo, "default_branch", None),
        "is_fork": bool(getattr(repo, "fork", False)),
        "language": getattr(repo, "language", None),
        "stargazers_count": getattr(repo, "stargazers_count", None),
        "open_issues_count": getattr(repo, "open_issues_count", None),
        "has_wiki": bool(getattr(repo, "has_wiki", False)),
        "created_at": getattr(repo, "created_at", None),
        "updated_at": getattr(repo, "updated_at", None),
        "pushed_at": getattr(repo, "pushed_at", None),
        "size_kb": getattr(repo, "size", None),
        "topics_csv": ",".join(getattr(repo, "get_topics", lambda: [])()),
    }

    return (repo_meta, signals, commits_by_user, issues_by_user, prs_by_user,
            team_weekly_commits, mature_issue_by_user, mature_pr_by_user,
            issue_comments_by_user_per_week, issue_type_diversity_by_user,
            atomicity_by_user, commit_messages_by_user, issue_texts_by_user,
            review_comments_by_user)

# ------------------ Scorer ------------------

def clamp01(x): return max(0.0, min(1.0, float(x)))

def fulfill_ratio(total, weeks, target_per_week):
    if weeks <= 0:
        return 0.0
    expected = target_per_week * weeks
    if expected <= 0:
        return 1.0
    return clamp01(total / expected)

def issue_type_diversity_score(counter: Counter):
    distinct = len([k for k,v in counter.items() if v > 0])
    return distinct / 4.0

def commit_curve_stability_score(team_weekly_commits: Counter, weeks):
    if weeks < STABILITY_MIN_WEEKS:
        return 0.5
    series = [team_weekly_commits.get(w, 0) for w in range(weeks)]
    mean = statistics.mean(series) if series else 0
    if mean == 0:
        return 0.5
    std = statistics.pstdev(series)
    rel = std / mean
    return clamp01(1 - math.tanh(rel))

def build_user_scorer(since_dt, signals,
                      commits_by_user, issues_by_user, prs_by_user,
                      team_weekly_commits,
                      mature_issue_by_user, mature_pr_by_user,
                      issue_comments_by_user_per_week,
                      issue_type_diversity_by_user,
                      atomicity_by_user,
                      commit_messages_by_user,
                      issue_texts_by_user,
                      review_comments_by_user,
                      ai_analyzer: Optional[AgileQualityAnalyzer] = None):
    weeks = max(1, math.ceil((dt.datetime.now(dt.timezone.utc) - since_dt).days / 7))
    oss_score = oss_signal_score(signals)

    def scorer(user):
        c = commits_by_user.get(user, 0)
        i = issues_by_user.get(user, 0)
        p = prs_by_user.get(user, 0)
        ic_week = issue_comments_by_user_per_week.get(user, Counter())

        r_commits = fulfill_ratio(c, weeks, WEEKLY_TARGETS["commits"])
        r_issues  = fulfill_ratio(i, weeks, WEEKLY_TARGETS["issues"])
        r_prs     = fulfill_ratio(p, weeks, WEEKLY_TARGETS["prs"])
        r_ic      = clamp01(sum(ic_week.values()) / max(1, weeks * WEEKLY_TARGETS["issue_comments"]))

        m_issue = clamp01(mature_issue_by_user.get(user, 0) / max(1, i))
        m_pr    = clamp01(mature_pr_by_user.get(user, 0) / max(1, p))

        atomic = clamp01(atomicity_by_user.get(user, 0.5))
        div = issue_type_diversity_score(issue_type_diversity_by_user.get(user, Counter()))
        stability = commit_curve_stability_score(team_weekly_commits, weeks)
        
        # AI quality scores
        ai_commit_quality = 0.5
        ai_issue_quality = 0.5
        ai_review_quality = 0.5
        
        if ai_analyzer and ai_analyzer.enabled:
            user_commits = commit_messages_by_user.get(user, [])
            user_issues = issue_texts_by_user.get(user, [])
            user_reviews = review_comments_by_user.get(user, [])
            
            if user_commits:
                ai_commit_quality = ai_analyzer.analyze_commit_quality(user_commits)
            if user_issues:
                ai_issue_quality = ai_analyzer.analyze_issue_quality(user_issues)
            if user_reviews:
                ai_review_quality = ai_analyzer.analyze_review_quality(user_reviews)

        s = 0.0
        s += CRITERIA_WEIGHTS["weekly_commits"] * r_commits
        s += CRITERIA_WEIGHTS["weekly_issues"]  * r_issues
        s += CRITERIA_WEIGHTS["weekly_prs"]     * r_prs
        s += CRITERIA_WEIGHTS["weekly_issue_comments"] * r_ic
        s += CRITERIA_WEIGHTS["mature_issue_ratio"] * m_issue
        s += CRITERIA_WEIGHTS["mature_pr_ratio"]    * m_pr
        s += CRITERIA_WEIGHTS["oss_signals"]        * oss_score
        s += CRITERIA_WEIGHTS["atomic_commits"]     * atomic
        s += CRITERIA_WEIGHTS["issue_type_diversity"] * div
        s += CRITERIA_WEIGHTS["commit_curve_stability"] * stability
        s += CRITERIA_WEIGHTS["ai_commit_quality"] * ai_commit_quality
        s += CRITERIA_WEIGHTS["ai_issue_quality"] * ai_issue_quality
        s += CRITERIA_WEIGHTS["ai_review_quality"] * ai_review_quality

        if s >= 0.75: nivel = "Maduro"
        elif s >= 0.45: nivel = "Saudável"
        elif s > 0.0:   nivel = "Iniciante"
        else:           nivel = "Sem evidências recentes"

        rationale = (
            f"sem={weeks} | commits {c} (r={r_commits:.2f}, ai_qual={ai_commit_quality:.2f}), "
            f"issues {i} (r={r_issues:.2f}, maduros={m_issue:.2f}, ai_qual={ai_issue_quality:.2f}), "
            f"PRs {p} (r={r_prs:.2f}, maduros={m_pr:.2f}), "
            f"coment/issue r={r_ic:.2f}, atomic={atomic:.2f}, "
            f"divIssues={div:.2f}, oss={oss_score:.2f}, stab={stability:.2f}, "
            f"ai_review={ai_review_quality:.2f}"
        )

        comps = {
            "r_commits": r_commits, "r_issues": r_issues, "r_prs": r_prs, "r_issue_comments": r_ic,
            "mature_issue_ratio": m_issue, "mature_pr_ratio": m_pr,
            "oss_signals": oss_score, "atomicity": atomic,
            "issue_type_diversity": div, "commit_curve_stability": stability,
            "ai_commit_quality": ai_commit_quality, "ai_issue_quality": ai_issue_quality,
            "ai_review_quality": ai_review_quality, "weeks_window": weeks,
        }
        return nivel, s, comps, rationale

    return scorer

# ------------------ Auth e CLI ------------------

def generate_evaluation_explanation(user: str, score: float, nivel: str, acc: dict, weeks_in_period: int = None) -> str:
    """Gera explicação geral da avaliação de um usuário."""
    explanations = []
    
    # Análise do nível geral
    if nivel == "Maduro":
        explanations.append("Demonstra excelentes práticas ágeis")
    elif nivel == "Saudável":
        explanations.append("Apresenta boas práticas com espaço para crescimento")
    elif nivel == "Iniciante":
        explanations.append("Práticas básicas, necessita desenvolvimento")
    else:
        explanations.append("Sem atividade suficiente para avaliação")
        return "; ".join(explanations)
    
    # Análise de frequência de commits (NOVO)
    commits_total = acc["commits_sum"]
    if weeks_in_period and weeks_in_period > 0:
        commits_per_week = commits_total / weeks_in_period
        
        # Estimativa de semanas sem commits (assumindo distribuição uniforme)
        if commits_total == 0:
            weeks_without_commits = weeks_in_period
        else:
            # Se tem menos de 1 commit por semana, estimar semanas inativas
            weeks_without_commits = max(0, weeks_in_period - commits_total)
        
        if commits_per_week < 1.0:
            if commits_total == 0:
                explanations.append(f"🚨 CRÍTICO: {weeks_in_period} semanas sem nenhum commit")
            elif weeks_without_commits >= weeks_in_period * 0.7:  # Mais de 70% das semanas sem commits
                explanations.append(f"⚠️ ALERTA: ~{weeks_without_commits:.0f} semanas inativas de {weeks_in_period} semanas")
            elif weeks_without_commits >= weeks_in_period * 0.4:  # Mais de 40% das semanas sem commits
                explanations.append(f"⚠️ Atenção: ~{weeks_without_commits:.0f} semanas com baixa atividade de {weeks_in_period} semanas")
            elif commits_per_week < 0.5:  # Menos de 0.5 commits por semana
                explanations.append(f"Frequência baixa: {commits_per_week:.1f} commits/semana")
    
    # Análise de atividade
    total_activity = acc["commits_sum"] + acc["issues_sum"] + acc["prs_sum"]
    if total_activity >= 50:
        explanations.append("alta atividade")
    elif total_activity >= 20:
        explanations.append("atividade moderada")
    elif total_activity >= 5:
        explanations.append("atividade baixa")
    else:
        explanations.append("atividade mínima")
    
    # Análise de qualidade
    f = lambda xs: sum(xs)/len(xs) if xs else 0.0
    atomicity_avg = f(acc["atomicity_vals"])
    oss_avg = f(acc["oss_vals"])
    
    if atomicity_avg >= 0.7:
        explanations.append("commits bem estruturados")
    elif atomicity_avg >= 0.4:
        explanations.append("commits razoáveis")
    else:
        explanations.append("commits precisam melhorar")
    
    if oss_avg >= 0.7:
        explanations.append("projetos bem documentados")
    elif oss_avg >= 0.4:
        explanations.append("documentação adequada")
    else:
        explanations.append("documentação insuficiente")
    
    # Análise de colaboração
    mature_ratio = (acc["mature_issues_sum"] + acc["mature_prs_sum"]) / max(1, acc["issues_sum"] + acc["prs_sum"])
    if mature_ratio >= 0.6:
        explanations.append("boa colaboração em discussões")
    elif mature_ratio >= 0.3:
        explanations.append("colaboração moderada")
    else:
        explanations.append("pouca participação em discussões")
    
    # Análise de IA (se disponível)
    ai_scores = [f(acc["ai_commit_vals"]), f(acc["ai_issue_vals"]), f(acc["ai_review_vals"])]
    ai_avg = sum([s for s in ai_scores if s > 0]) / max(1, len([s for s in ai_scores if s > 0]))
    
    if ai_avg >= 0.7:
        explanations.append("qualidade de comunicação excelente")
    elif ai_avg >= 0.5:
        explanations.append("qualidade de comunicação boa")
    elif ai_avg > 0:
        explanations.append("qualidade de comunicação pode melhorar")
    
    return "; ".join(explanations)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Organização do GitHub")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (início da janela)")
    ap.add_argument("--out", default="avaliacao.xlsx", help="Arquivo XLSX de saída")
    ap.add_argument("--token", default=None, help="GitHub Token (ou via GH_TOKEN env)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--only-recent", action="store_true", help="pular repos com pushed_at muito antigo (> 1 ano)")
    ap.add_argument("--include-new-repos", action="store_true", help="incluir repositórios criados a partir da data --since")
    ap.add_argument("--users-csv", default=None, help="CSV com coluna github_username para filtrar usuários de interesse")
    ap.add_argument("--debug-auth", action="store_true", help="Imprime usuário autenticado e dicas se falhar")
    ap.add_argument("--openai-key", default=None, help="OpenAI API Key (ou via OPENAI_API_KEY env)")
    ap.add_argument("--disable-ai", action="store_true", help="Desabilita análise AI mesmo com chave disponível")
    return ap.parse_args()

def debug_auth_or_die(gh: Github, org_name: str, debug: bool):
    try:
        # Teste rápido de /user (funciona só autenticado)
        me = gh.get_user()
        _ = me.login  # força request
        if debug:
            print(f"[auth] autenticado como: {me.login}")
    except GithubException as e:
        if debug:
            print("[auth] Falha ao obter /user. Possíveis causas: token vazio/revogado; fine-grained sem owner; SSO não autorizado.")
        raise

    # Antes de chamar /orgs/<org>, cheque rate
    rate_limit_guard(gh, threshold=50)
    try:
        org = gh.get_organization(org_name)
        return org
    except GithubException as e:
        if e.status == 401:
            msg = (
                "\n[auth] 401 Bad credentials.\n"
                " Verifique:\n"
                "  1) GH_TOKEN exportado sem aspas/linhas novas (print(repr(...))).\n"
                "  2) Se token é fine-grained: Resource owner = a organização; repo access e permissões Read (contents, metadata, issues, PRs, members). Autorize SSO.\n"
                "  3) Se token é clássico: escopos repo + read:org. Autorize SSO na org.\n"
                "  4) Teste: curl -H 'Authorization: Bearer $GH_TOKEN' https://api.github.com/user\n"
            )
            print(msg, file=sys.stderr)
        raise

# ------------------ MAIN ------------------

def main():
    args = parse_args()
    # Load .env if available (before reading env vars)
    if 'load_dotenv' in globals() and DOTENV_AVAILABLE:
        load_dotenv()
    token = (args.token or os.getenv("GH_TOKEN", "")).strip()
    gh = Github(token, per_page=50) if token else Github(per_page=50)
    
    # Initialize AI analyzer
    ai_analyzer = None
    if not args.disable_ai:
        ai_analyzer = AgileQualityAnalyzer(args.openai_key)
        if ai_analyzer.enabled:
            print("[ai] AI-powered quality analysis enabled")
        else:
            print("[ai] AI analysis disabled (no API key or initialization failed)")

    try:
        core = gh.get_rate_limit().core
        print(f"[rate] start remaining={core.remaining}, reset={core.reset}")
    except Exception:
        pass

    since_dt = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

    try:
        org = debug_auth_or_die(gh, args.org, args.debug_auth)
    except GithubException as e:
        # Mostra a mensagem original também
        print(f"[error] GitHub API: {e.status} {getattr(e, 'data', '')}", file=sys.stderr)
        sys.exit(1)

    target_users = None
    if args.users_csv:
        dfu = pd.read_csv(args.users_csv)
        target_users = set([u.lower() for u in dfu["github_username"].astype(str).tolist()])

    prefiltered = []
    for r in org.get_repos():
        if args.skip_forks and getattr(r, "fork", False):
            continue
        
        # Filtro por atividade recente (pushed_at)
        if args.only_recent and getattr(r, "pushed_at", dt.datetime(1970,1,1, tzinfo=dt.timezone.utc)) < since_dt:
            continue
        
        # Filtro por data de criação (se habilitado)
        if args.include_new_repos:
            repo_created_at = getattr(r, "created_at", dt.datetime(1970,1,1, tzinfo=dt.timezone.utc))
            repo_pushed_at = getattr(r, "pushed_at", dt.datetime(1970,1,1, tzinfo=dt.timezone.utc))
            
            # Incluir se: criado após since_dt OU teve atividade após since_dt
            if repo_created_at >= since_dt or repo_pushed_at >= since_dt:
                prefiltered.append(r)
        else:
            # Comportamento padrão: apenas por atividade
            prefiltered.append(r)

    rows_details = []
    user_accum = defaultdict(lambda: {
        "scores": [],
        "commits_sum": 0, "issues_sum": 0, "prs_sum": 0,
        "mature_issues_sum": 0, "mature_prs_sum": 0,
        "issue_comments_sum": 0,
        "atomicity_vals": [],
        "diversity_vals": [],
        "oss_vals": [],
        "stability_vals": [],
        "ai_commit_vals": [],
        "ai_issue_vals": [],
        "ai_review_vals": [],
        "rationales": [],
    })

    try:
        core = gh.get_rate_limit().core
        dyn_workers = max(2, min(args.workers, int(core.remaining // 100) or 2))
        print(f"[pool] usando {dyn_workers} workers (req remaining ~{core.remaining})")
    except Exception:
        dyn_workers = args.workers

    with ThreadPoolExecutor(max_workers=dyn_workers) as ex:
        futures = {ex.submit(collect_repo_contrib, r, since_dt, ai_analyzer): r.full_name for r in prefiltered}
        for fut in as_completed(futures):
            rate_limit_guard(gh, threshold=100)
            repo_name = futures[fut]
            try:
                (repo_meta, signals, commits_by_user, issues_by_user, prs_by_user,
                 team_weekly_commits, mature_issue_by_user, mature_pr_by_user,
                 issue_comments_by_user_per_week, issue_type_diversity_by_user,
                 atomicity_by_user, commit_messages_by_user, issue_texts_by_user,
                 review_comments_by_user) = fut.result()
            except Exception as e:
                print(f"[warn] {repo_name} failed: {e}")
                continue

            scorer = build_user_scorer(
                since_dt, signals,
                commits_by_user, issues_by_user, prs_by_user,
                team_weekly_commits, mature_issue_by_user, mature_pr_by_user,
                issue_comments_by_user_per_week, issue_type_diversity_by_user,
                atomicity_by_user, commit_messages_by_user, issue_texts_by_user,
                review_comments_by_user, ai_analyzer
            )

            seen_users = set(commits_by_user) | set(issues_by_user) | set(prs_by_user) | set(issue_comments_by_user_per_week)
            for user in seen_users:
                if target_users and user not in target_users:
                    continue

                nivel, score, comps, rationale = scorer(user)
                c = commits_by_user.get(user, 0)
                i = issues_by_user.get(user, 0)
                p = prs_by_user.get(user, 0)
                mi = mature_issue_by_user.get(user, 0)
                mp = mature_pr_by_user.get(user, 0)
                ic = sum(issue_comments_by_user_per_week.get(user, Counter()).values())
                atomic = comps["atomicity"]
                div = comps["issue_type_diversity"]
                oss = comps["oss_signals"]
                stab = comps["commit_curve_stability"]
                ai_cq = comps.get("ai_commit_quality", 0.5)
                ai_iq = comps.get("ai_issue_quality", 0.5)
                ai_rq = comps.get("ai_review_quality", 0.5)

                user_accum[user]["scores"].append(score)
                user_accum[user]["commits_sum"] += c
                user_accum[user]["issues_sum"]  += i
                user_accum[user]["prs_sum"]     += p
                user_accum[user]["mature_issues_sum"] += mi
                user_accum[user]["mature_prs_sum"]    += mp
                user_accum[user]["issue_comments_sum"] += ic
                user_accum[user]["atomicity_vals"].append(atomic)
                user_accum[user]["diversity_vals"].append(div)
                user_accum[user]["oss_vals"].append(oss)
                user_accum[user]["stability_vals"].append(stab)
                user_accum[user]["ai_commit_vals"].append(ai_cq)
                user_accum[user]["ai_issue_vals"].append(ai_iq)
                user_accum[user]["ai_review_vals"].append(ai_rq)
                user_accum[user]["rationales"].append(f"[{repo_name}] {rationale}")

                rows_details.append({
                    "github_username": user, "since_date": args.since, **repo_meta,
                    "commits_recent": c, "issues": i, "prs": p,
                    "mature_issues": mi, "mature_prs": mp, "issue_comments": ic,
                    "has_readme": signals["has_readme"], "has_license": signals["has_license"],
                    "has_contributing": signals["has_contributing"], "has_dev_env": signals["has_dev_env"],
                    "has_tests": signals["has_tests"], "has_workflows": signals["has_workflows"],
                    "has_coverage": signals["has_coverage"], "has_deploy_hints": signals["has_deploy_hints"],
                    "has_readme_with_run": signals["has_readme_with_run"],
                    "r_commits_repo": round(comps["r_commits"], 3), "r_issues_repo": round(comps["r_issues"], 3),
                    "r_prs_repo": round(comps["r_prs"], 3), "r_issue_comments_repo": round(comps["r_issue_comments"], 3),
                    "mature_issue_ratio_repo": round(comps["mature_issue_ratio"], 3),
                    "mature_pr_ratio_repo": round(comps["mature_pr_ratio"], 3),
                    "atomicity_repo": round(comps["atomicity"], 3),
                    "issue_type_diversity_repo": round(comps["issue_type_diversity"], 3),
                    "oss_signals_repo": round(comps["oss_signals"], 3),
                    "commit_curve_stability_repo": round(comps["commit_curve_stability"], 3),
                    "ai_commit_quality_repo": round(ai_cq, 3),
                    "ai_issue_quality_repo": round(ai_iq, 3),
                    "ai_review_quality_repo": round(ai_rq, 3),
                    "score_repo": round(score, 3), "nivel_repo": nivel, "rationale_repo": rationale,
                })

    summary_rows = []
    users_iter = sorted(target_users) if target_users else sorted(user_accum.keys())
    for user in users_iter:
        acc = user_accum.get(user)
        if not acc or not acc["scores"]:
            summary_rows.append({
                "github_username": user, "score_final_0_1": 0.0, "nivel": "Sem evidências recentes",
                "explicacao_geral": "Usuário sem atividade suficiente no período analisado",
                "commits_total": 0, "issues_total": 0, "prs_total": 0,
                "mature_issues_total": 0, "mature_prs_total": 0, "issue_comments_total": 0,
                "atomicity_media": 0.0, "issue_type_diversity_media": 0.0,
                "oss_signals_media": 0.0, "commit_curve_stability_media": 0.0,
                "ai_commit_quality_media": 0.0, "ai_issue_quality_media": 0.0, "ai_review_quality_media": 0.0,
                "rationale": "", "rituals_participation": "", "sprint_retro_regression": "", "scrum_metrics_review": "",
            })
            continue

        f = lambda xs: sum(xs)/len(xs) if xs else 0.0
        final = f(acc["scores"])
        if final >= 0.75: nivel = "Maduro"
        elif final >= 0.45: nivel = "Saudável"
        else: nivel = "Iniciante"

        # Calcular número de semanas no período
        now_dt = dt.datetime.now(dt.timezone.utc)
        weeks_in_period = max(1, (now_dt - since_dt).days // 7)
        
        # Gerar explicação geral
        explicacao = generate_evaluation_explanation(user, final, nivel, acc, weeks_in_period)

        summary_rows.append({
            "github_username": user, "score_final_0_1": round(final, 3), "nivel": nivel,
            "explicacao_geral": explicacao,
            "commits_total": acc["commits_sum"], "issues_total": acc["issues_sum"], "prs_total": acc["prs_sum"],
            "mature_issues_total": acc["mature_issues_sum"], "mature_prs_total": acc["mature_prs_sum"],
            "issue_comments_total": acc["issue_comments_sum"],
            "atomicity_media": round(f(acc["atomicity_vals"]), 3),
            "issue_type_diversity_media": round(f(acc["diversity_vals"]), 3),
            "oss_signals_media": round(f(acc["oss_vals"]), 3),
            "commit_curve_stability_media": round(f(acc["stability_vals"]), 3),
            "ai_commit_quality_media": round(f(acc["ai_commit_vals"]), 3),
            "ai_issue_quality_media": round(f(acc["ai_issue_vals"]), 3),
            "ai_review_quality_media": round(f(acc["ai_review_vals"]), 3),
            "rationale": " | ".join(acc["rationales"][-3:]),
            "rituals_participation": "", "sprint_retro_regression": "", "scrum_metrics_review": "",
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "score_final_0_1" in summary_df.columns:
        summary_df = summary_df.sort_values("score_final_0_1", ascending=False)
    details_df = pd.DataFrame(rows_details)
    needed_cols = ["github_username", "repo_full_name", "since_date"]
    if not details_df.empty and all(c in details_df.columns for c in needed_cols):
        details_df = details_df.sort_values(needed_cols)
    provenance_df = pd.DataFrame([{
        "org": args.org, "since": args.since, "workers": args.workers,
        "skip_forks": args.skip_forks, "only_recent": args.only_recent, "include_new_repos": args.include_new_repos,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(),
        "script": "aval-mds.py", "version_hint": "v4.4 – criterios + rate-limit + auth diag + OpenAI integration + filtro por data de criação + explicação geral + suporte co-authors + alerta frequência commits",
        "weights": CRITERIA_WEIGHTS, "weekly_targets": WEEKLY_TARGETS,
    }])

    # Remove timezone info from datetime columns to avoid Excel errors
    for df in [summary_df, details_df, provenance_df]:
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns, UTC]' or 'datetime' in str(df[col].dtype):
                try:
                    df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
                except:
                    pass

    with pd.ExcelWriter(args.out, engine="openpyxl") as xlw:
        summary_df.to_excel(xlw, index=False, sheet_name="Resumo_por_usuario")
        details_df.to_excel(xlw, index=False, sheet_name="Detalhes_por_repo")
        provenance_df.to_excel(xlw, index=False, sheet_name="Proveniencia")

    print(f"OK: {args.out}")

if __name__ == "__main__":
    main()
