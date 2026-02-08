import requests

def get_gitee_pr_with_diff(owner, repo, pr_number):
    """
    一站式获取 Gitee PR 信息及对应的代码 diff 内容
    :param owner: 仓库拥有者（用户名/组织名）
    :param repo: 仓库名称
    :param pr_number: PR 编号（如 1）
    :return: 字典，包含 PR 基础信息 + diff 内容；失败返回 None
    """
    # 第一步：获取 PR 基础信息
    pr_info_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}"
    try:
        # 请求 PR 基础信息
        pr_response = requests.get(pr_info_url)
        pr_response.raise_for_status()
        pr_info = pr_response.json()
        
        # 第二步：通过 diff_url 获取代码 diff 内容
        diff_url = pr_info.get('diff_url')
        if not diff_url:
            pr_info['diff_content'] = ""
            return pr_info
        
        diff_response = requests.get(diff_url)
        diff_response.raise_for_status()
        # 处理编码，避免中文乱码
        diff_content = diff_response.text.encode('utf-8').decode('utf-8', 'ignore')
        
        # 将 diff 内容整合到 PR 信息字典中
        pr_info['diff_content'] = diff_content
        
        print(pr_info['diff_content'])
        # print(f"PR 标题：{pr_info['title']}")
        # print(f"PR 状态：{pr_info['state']}")
        # print(f"PR 链接：{pr_info['html_url']}")
        # print("\n=== 代码 Diff 内容 ===")
        # print(pr_info['diff_content'])
        return pr_info['diff_content']
        
    except requests.exceptions.RequestException as e:
        print(f"获取 PR 信息或 diff 失败：{e}")
        return None

# 外部调用示例
if __name__ == "__main__":
    # 配置目标仓库和 PR 编号
    owner = "poppyaxelord"
    repo = "Langgraph-task-split"
    pr_number = 1
    
    # 一键调用，获取包含 diff 的完整 PR 信息
    pr_data = get_gitee_pr_with_diff(owner, repo, pr_number)
