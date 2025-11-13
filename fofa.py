#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
import asyncio
import httpx  # 用于异步 HTTP 请求
import base64
import yaml    # PyYAML
import re      # 正则表达式
import warnings
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple, Any, Literal, TypedDict
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from pyfiglet import Figlet

load_dotenv()


# --- 常量配置 ---
# (这些现在会自动从 .env 文件读取)
CONCURRENCY_LIMIT = int(os.getenv('CONCURRENCY_LIMIT', '5'))
FOFA_KEY = os.getenv('FOFA_KEY')
FOFA_SIZE = int(os.getenv('FOFA_SIZE', '20'))
# 忽略 httpx 的 SSL 验证警告
# 原始脚本中 { tls: { rejectUnauthorized: false } } 相当于 verify=False
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# --- 类型定义 (使用 Type Hints 和 TypedDict) ---
class FofaTarget(TypedDict):
    host: str
    protocol: str
    header: str
    banner: str

class PageResult(TypedDict):
    host: str
    body: str
    header: str
    banner: str

class SubscriptionUserinfo(TypedDict):
    upload: int
    download: int
    total: int
    expire: Optional[int]

VerificationStatus = Literal['success', 'failed']

class VerificationResult(TypedDict):
    link: str
    host: str
    status: VerificationStatus
    reason: Optional[str]

class LinkToVerify(TypedDict):
    link: str
    host: str

# --- 常量配置 ---
CONCURRENCY_LIMIT = int(os.getenv('CONCURRENCY_LIMIT', '5'))
REQUEST_TIMEOUT = 5.0  # httpx 使用浮点数
FOFA_SEARCH_PATH = "/api/v1/search/all"
# Python 的 re.compile
SUBSCRIPTION_REGEX = re.compile(r'(https?:\/\/[^\s\"\'<>`]+\/api\/v1\/client\/subscribe\?token=[a-zA-Z0-9]+)')
UNITS = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
UNIT_POWERS = [1024**i for i in range(len(UNITS))]

# --- Fofa API 配置 ---
FOFA_KEY = os.getenv('FOFA_KEY')
FOFA_FIELDS = "host,protocol,header,banner"
FOFA_SIZE = int(os.getenv('FOFA_SIZE', '20'))

# --- UI 与日志工具 ---
console = Console()

class Logger:
    @staticmethod
    def step(step: int, total: int, title: str):
        console.print(f"\n--- Step {step}/{total}: {title} ---", style="bold white")

    @staticmethod
    def info(message: str):
        console.print(message, style="magenta")

    @staticmethod
    def success(message: str):
        console.print(message, style="green")

    @staticmethod
    def error(message: str):
        console.print(f"错误：{message}", style="bold red")

    @staticmethod
    def warning(message: str):
        console.print(message, style="yellow")

    @staticmethod
    def cyan(message: str):
        console.print(message, style="cyan")

logger = Logger()

# --- 配置验证 ---
def validate_config():
    if not FOFA_KEY:
        logger.error("请配置环境变量 FOFA_KEY。")
        console.print("您可以从 https://fofoapi.com/userInfo 获取您的key")
        sys.exit(1)
    
    if FOFA_SIZE < 1:
        logger.error("FOFA_SIZE 必须大于 0。")
        sys.exit(1)
    
    if CONCURRENCY_LIMIT < 1:
        logger.error("CONCURRENCY_LIMIT 必须大于 0。")
        sys.exit(1)

# --- 辅助函数 ---
CLASH_UA = 'clash'

def parse_subscription_userinfo(header_value: Optional[str]) -> Optional[SubscriptionUserinfo]:
    if not header_value:
        return None
    
    result: SubscriptionUserinfo = {'upload': 0, 'download': 0, 'total': 0, 'expire': None}
    
    try:
        pairs = [pair.strip() for pair in header_value.split(';')]
        for pair in pairs:
            parts = pair.split('=')
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), parts[1].strip()
            
            if value:
                num_value = int(value)
                if key == 'upload':
                    result['upload'] = num_value
                elif key == 'download':
                    result['download'] = num_value
                elif key == 'total':
                    result['total'] = num_value
                elif key == 'expire':
                    result['expire'] = num_value
    except (ValueError, IndexError):
        logger.warning(f"Failed to parse subscription-userinfo: {header_value}")
        return None  # 解析出错
    
    return result

def parse_traffic(num: Optional[int]) -> Tuple[str, str]:
    if not isinstance(num, (int, float)) or num < 0:
        return "NaN", ""
    if num < 1000:
        return f"{round(num)}", "B"
    
    # 使用 log2 来计算更精确，但 bit_length 更简单
    if num == 0: return "0", "B"
    exp = min(max(0, int((num.bit_length() - 1) / 10)), len(UNITS) - 1)
    dat = num / (UNIT_POWERS[exp] or 1)
    ret = f"{dat:.0f}" if dat >= 1000 else f"{dat:.3g}"
    unit = UNITS[exp] or 'B'
    return ret, unit

def calculate_used_traffic(userinfo: SubscriptionUserinfo) -> int:
    return userinfo['upload'] + userinfo['download']

def format_usage_info(userinfo: SubscriptionUserinfo) -> str:
    used = calculate_used_traffic(userinfo)
    
    def format_bytes(bytes_val: int) -> str:
        value, unit = parse_traffic(bytes_val)
        return f"{value} {unit}"
    
    total_str = format_bytes(userinfo['total']) if userinfo['total'] > 0 else "Unlimited"
    info = f"{format_bytes(used)}/{total_str}"
    
    if userinfo['expire']:
        try:
            expire_date = datetime.fromtimestamp(userinfo['expire'])
            now = datetime.now()
            
            if expire_date > now:
                formatted_date = expire_date.strftime('%Y-%m-%d')
                info += f" ({formatted_date})"
            else:
                info += " (已过期)"
        except Exception:
            info += " (日期无效)"
    
    return info

def validate_subscription(userinfo: SubscriptionUserinfo) -> Tuple[bool, Optional[str]]:
    used = calculate_used_traffic(userinfo)
    
    # 只有当 total > 0 时才检查流量
    if userinfo['total'] > 0 and used >= userinfo['total']:
        return False, '流量已用完'
    
    if userinfo['expire']:
        now = int(datetime.now().timestamp())
        if userinfo['expire'] <= now:
            return False, '已过期'
    
    return True, None

def parse_yaml_content(body: str) -> bool:
    try:
        # 使用 safe_load 防止任意代码执行
        parsed = yaml.safe_load(body)
        
        if not parsed or not isinstance(parsed, dict):
            return False
        
        if 'proxy-groups' not in parsed or not isinstance(parsed['proxy-groups'], list):
            return False
        
        if len(parsed['proxy-groups']) == 0:
            return False
            
        return True
    except yaml.YAMLError:
        return False

# --- 核心功能函数 (Async) ---

async def query_fofa_api_generic(
    client: httpx.AsyncClient, 
    query_string: str, 
    description: str
) -> List[FofaTarget]:
    logger.info(f"Starting Fofa API query for {description}...")
    
    query_base64 = base64.b64encode(query_string.encode()).decode()
    fofa_url = f"https://fofoapi.com{FOFA_SEARCH_PATH}"
    params = {
        'key': FOFA_KEY,
        'qbase64': query_base64,
        'fields': FOFA_FIELDS,
        'size': str(FOFA_SIZE) # Fofa API 期望 size 是字符串
    }
    
    try:
        response = await client.get(fofa_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # 检查 HTTP 4xx/5xx 错误
        
        fofa_data = response.json()
        
        if fofa_data.get('error'):
            raise Exception(f"Fofa API 错误: {fofa_data.get('errmsg', 'Unknown error')}")
        
        results = fofa_data.get('results')
        if not results:
            return []
            
        logger.success(f"Fofa API query for {description} completed.")
        
        # TS: r: [string, string, string, string]
        return [
            {
                'host': r[0],
                'protocol': r[1],
                'header': r[2] or '',
                'banner': r[3] or ''
            } 
            for r in results
        ]
    except httpx.HTTPStatusError as e:
        raise Exception(f"Fofa API request failed with status: {e.response.status_code}")
    except Exception as e:
        raise Exception(f"Fofa query failed: {e}")

async def query_fofa_api(client: httpx.AsyncClient) -> List[FofaTarget]:
    subscription_token_query = "/api/v1/client/subscribe?token="
    return await query_fofa_api_generic(client, subscription_token_query, "subscription token search")

async def query_fofa_api_for_subscription_headers(client: httpx.AsyncClient) -> List[FofaTarget]:
    subscription_query = 'header="subscription-userinfo" || banner="subscription-userinfo"'
    return await query_fofa_api_generic(client, subscription_query, "subscription-userinfo headers")

async def fetch_page_content_worker(
    client: httpx.AsyncClient, 
    target: FofaTarget, 
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: Any
) -> Optional[PageResult]:
    async with semaphore:
        final_url = target['host'] if target['host'].startswith('http') else f"{target.get('protocol', 'http')}://{target['host']}"
        try:
            res = await client.get(final_url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            if not res.is_success:
                return None
            
            # 尝试 utf-8 解码，失败则回退到 latin-1
            try:
                body_text = res.content.decode('utf-8')
            except UnicodeDecodeError:
                body_text = res.content.decode('latin-1', errors='ignore')

            return {
                'host': target['host'],
                'body': body_text,
                'header': target['header'], # 这是 Fofa 缓存的 header
                'banner': target['banner']  # 这是 Fofa 缓存的 banner
            }
        except Exception:
            return None
        finally:
            progress.update(task_id, advance=1)

async def fetch_page_contents(client: httpx.AsyncClient, targets: List[FofaTarget]) -> List[PageResult]:
    logger.info(f"Fetching page content from {len(targets)} targets (Concurrency: {CONCURRENCY_LIMIT})")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    with Progress(
        TextColumn("  fetching [progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed} of {task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("pages", total=len(targets))
        for target in targets:
            tasks.append(fetch_page_content_worker(client, target, semaphore, progress, task_id))
        
        results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r]
    logger.info(f"\nPage content fetched. Found {len(valid_results)} valid pages.")
    return valid_results

def extract_subscription_links(page_results: List[PageResult]) -> List[LinkToVerify]:
    logger.info(f"Processing {len(page_results)} pages to extract subscription links...")
    unique_potential_links: Dict[str, str] = {}  # 使用 dict 作为 Map
    
    def find_and_add_links(content: str, host: str):
        if not content:
            return
        # 使用 finditer 来查找所有匹配项
        matches = SUBSCRIPTION_REGEX.finditer(content)
        for match in matches:
            link = match.group(0)
            if link not in unique_potential_links:
                unique_potential_links[link] = host
    
    for pr in page_results:
        find_and_add_links(pr['body'], pr['host'])
        find_and_add_links(pr['header'], pr['host'])
        find_and_add_links(pr['banner'], pr['host'])
    
    potential_links = [{'link': link, 'host': host} for link, host in unique_potential_links.items()]
    
    if potential_links:
        logger.info(f"Extracted {len(potential_links)} unique potential links.")
    
    return potential_links

def process_subscription_hosts(subscription_targets: List[FofaTarget]) -> List[LinkToVerify]:
    logger.info(f"Processing {len(subscription_targets)} subscription service hosts...")
    links: List[LinkToVerify] = []
    for target in subscription_targets:
        final_url = target['host'] if target['host'].startswith('http') else f"{target.get('protocol', 'http')}://{target['host']}"
        links.append({'link': final_url, 'host': target['host']})
    
    if links:
        logger.info(f"Generated {len(links)} subscription links from direct hosts.")
    
    return links

async def verify_link_worker(
    client: httpx.AsyncClient,
    item: LinkToVerify,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: Any
) -> VerificationResult:
    link, host = item['link'], item['host']
    
    async with semaphore:
        try:
            headers = {'User-Agent': CLASH_UA}
            res = await client.get(link, headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            
            if not res.is_success:
                return {'link': link, 'host': host, 'status': 'failed', 'reason': f"HTTP {res.status_code}"}
            
            # HTTP 头部是大小写不敏感的
            userinfo_header = res.headers.get('subscription-userinfo')
            
            if not userinfo_header:
                return {'link': link, 'host': host, 'status': 'failed', 'reason': '缺少 subscription-userinfo 响应头'}
            
            userinfo = parse_subscription_userinfo(userinfo_header)
            if not userinfo:
                return {'link': link, 'host': host, 'status': 'failed', 'reason': 'subscription-userinfo 解析失败'}
            
            # 尝试解码
            try:
                sub_body = res.content.decode('utf-8')
            except UnicodeDecodeError:
                sub_body = res.content.decode('latin-1', errors='ignore')
            
            if not parse_yaml_content(sub_body):
                return {'link': link, 'host': host, 'status': 'failed', 'reason': '非 YAML 内容'}
            
            is_valid, reason = validate_subscription(userinfo)
            if not is_valid:
                return {'link': link, 'host': host, 'status': 'failed', 'reason': reason}
            
            return {
                'link': link,
                'host': host,
                'status': 'success',
                'reason': format_usage_info(userinfo)
            }
        
        except httpx.TimeoutException:
            reason = "访问超时"
        except httpx.RequestError as e:
            reason = f"访问失败 ({type(e).__name__})"
        except Exception as e:
            reason = f"未知错误 ({e})"
        finally:
            progress.update(task_id, advance=1)
        
        return {'link': link, 'host': host, 'status': 'failed', 'reason': reason}

async def verify_subscription_links(
    client: httpx.AsyncClient, 
    links_to_verify: List[LinkToVerify]
) -> List[VerificationResult]:
    logger.info(f"Verifying {len(links_to_verify)} potential links (Concurrency: {CONCURRENCY_LIMIT})")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    with Progress(
        TextColumn("  verifying [progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed} of {task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("links", total=len(links_to_verify))
        for item in links_to_verify:
            tasks.append(verify_link_worker(client, item, semaphore, progress, task_id))
        
        results = await asyncio.gather(*tasks)
    
    logger.info("\nLink verification completed.")
    return [r for r in results if r]

# --- (请使用这个新的 report_results 函数) ---

def report_results(results: List[VerificationResult]):
    logger.info("Reporting results...")
    
    successful_links = [r for r in results if r['status'] == 'success']
    
    file_content = []
    
    if successful_links:
        logger.success(f"\n[+] 发现 {len(successful_links)} 个有效的订阅链接:")
        
        # --- (*** 唯一的修改点在这个循环内部 ***) ---
        for r in successful_links:
            # 1. (不变) 准备来源 URL
            source_url = r['host'] if r['host'].startswith('http') else f"http://{r['host']}"
            
            # 2. (不变) 依旧打印到控制台
            console.print(f"  - {r['link']} (来源: {source_url})")
            
            # 3. (*** 修改 ***) 
            #    现在文件行也包含 "来源"
            file_line = r['link']
            file_line += f" | 来源: {source_url}"  # <--- 这是新增的部分

            if r['reason']:
                # 4. (不变) 依旧打印到控制台
                logger.cyan(f"    用量信息: {r['reason']}")
                
                # 5. (不变) 添加用量信息到文件行
                file_line += f" | 用量信息: {r['reason']}"
            
            file_content.append(file_line)
        # --- (*** 修改结束 ***) ---

        try:
            # (以下文件保存逻辑保持不变)
            now = datetime.now()
            output_dir = "result"
            os.makedirs(output_dir, exist_ok=True)
            
            base_filename = now.strftime("%Y%m%d%H%M") + ".txt"
            filename = os.path.join(output_dir, base_filename)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"--- F-Proxy 扫描结果 ---\n")
                f.write(f"--- 保存时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"共发现 {len(successful_links)} 个有效订阅:\n\n")
                
                for line in file_content:
                    f.write(line + '\n')
            
            logger.success(f"\n[+] 结果已成功保存到文件: {filename}")
            
        except Exception as e:
            logger.error(f"\n[!] 结果保存到文件失败: {e}")

    logger.info("----------------------------------------")
    if not successful_links:
        console.print("Task completed. No valid subscription links found.")
    else:
        console.print(f"Task completed! Found {len(successful_links)} valid subscription links.")
    logger.info("----------------------------------------")

def deduplicate_links(combined_links: List[LinkToVerify]) -> List[LinkToVerify]:
    unique_links_map: Dict[str, LinkToVerify] = {}
    
    for item in combined_links:
        try:
            url = urlparse(item['link'])
            # 使用 hostname + path + query 作为键
            host_key = (url.hostname or '') + (url.path or '') + (url.query or '')
            
            existing = unique_links_map.get(host_key)
            if not existing:
                unique_links_map[host_key] = item
            # 优先保留 HTTPS
            elif item['link'].startswith('https://') and existing['link'].startswith('http://'):
                unique_links_map[host_key] = item
        except Exception:
            # URL 解析失败，使用原始链接作为 key
            if item['link'] not in unique_links_map:
                    unique_links_map[item['link']] = item
    
    return list(unique_links_map.values())

# --- 主函数 ---
async def main():
    f = Figlet(font='block')
    # console.print(f.renderText('F-Proxy'), style="gradient(green,blue)")
    console.print(f.renderText('F-Proxy'), style="bold green")
    
    try:
        # 0. 验证配置
        validate_config()
        
        # 原始脚本禁用了 SSL 验证，这里也一样
        async with httpx.AsyncClient(verify=False) as client:
            
            # 1. 查询 Fofa (页面)
            logger.step(1, 6, "Querying Fofa for pages containing subscription links")
            fofa_targets = await query_fofa_api(client)
            if not fofa_targets:
                logger.warning("Fofa API (query 1) returned no results.")
            logger.success("--- Step 1/6 Completed ---")
            
            # 2. 查询 Fofa (Header)
            logger.step(2, 6, "Querying Fofa for subscription service hosts")
            subscription_targets = await query_fofa_api_for_subscription_headers(client)
            if not subscription_targets:
                logger.warning("Fofa API (query 2) returned no results.")
            logger.success("--- Step 2/6 Completed ---")

            # 如果两个查询都没有结果，则退出
            if not fofa_targets and not subscription_targets:
                 logger.warning("Both Fofa queries returned no results. Exiting.")
                 return

            # 3. 获取页面内容
            page_results: List[PageResult] = []
            if fofa_targets:
                logger.step(3, 6, "Fetching page contents")
                page_results = await fetch_page_contents(client, fofa_targets)
                if not page_results:
                    logger.warning("No pages could be fetched or processed.")
                logger.success("--- Step 3/6 Completed ---")
            else:
                logger.step(3, 6, "Fetching page contents (Skipped)")
                logger.warning("No targets from step 1, skipping page fetch.")
                logger.success("--- Step 3/6 Completed ---")
            
            # 4. 提取链接
            logger.step(4, 6, "Extracting subscription links")
            potential_links_to_verify = extract_subscription_links(page_results)
            subscription_links_to_verify = process_subscription_hosts(subscription_targets)
            
            combined_links = potential_links_to_verify + subscription_links_to_verify
            
            if not combined_links:
                logger.info("----------------------------------------")
                logger.warning("No potential subscription links found from any source.")
                return
            
            all_links_to_verify = deduplicate_links(combined_links)
            duplicate_count = len(combined_links) - len(all_links_to_verify)
            
            logger.info(f"Total links before deduplication: {len(combined_links)} ({len(potential_links_to_verify)} from body extraction + {len(subscription_links_to_verify)} from direct subscription hosts)")
            if duplicate_count > 0:
                logger.info(f"Removed {duplicate_count} duplicate links")
            logger.info(f"Final links to verify: {len(all_links_to_verify)}")
            logger.success("--- Step 4/6 Completed ---")
            
            # 5. 验证链接
            logger.step(5, 6, "Verifying subscription links")
            verification_results = await verify_subscription_links(client, all_links_to_verify)
            logger.success("--- Step 5/6 Completed ---")
            
            # 6. 报告结果
            logger.step(6, 6, "Reporting results")
            report_results(verification_results)
            logger.success("--- Step 6/6 Completed ---")
            
    except Exception as e:
        logger.error(f"\n处理过程中发生严重错误: {e}")
        # 打印完整的堆栈跟踪
        console.print_exception(show_locals=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nUser interrupted. Exiting.")
        sys.exit(0)