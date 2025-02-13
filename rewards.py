"""Reward functions for GRPO training."""
import re
from math_verify import  parse, verify
from langdetect import detect

def language_consistency_reward(completions, languages, **kwargs):
    """优化后的语言一致性奖励函数，整合所有功能并提升性能"""
    
    def _clean_content(text):
        """整合清理逻辑：移除标记和代码数学公式"""
        # 清理特殊标记
        text = re.sub(r'<[\/]?(thinking|output)[^>]*>', '', text)
        # 移除代码块（合并正则表达式）
        text = re.sub(r'(```[\s\S]*?```|`[^`]*?`|\$[\s\S]*?\$)', '', text)
        return text.strip()

    rewards = []
    for completion, lang in zip(completions, languages):
        try:
            content = completion[0]["content"].replace("<|im_end|>", "")
            cleaned = _clean_content(content)
            detected_lang = detect(cleaned)
            rewards.append(1.0 if detected_lang==lang else 0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def accuracy_reward(completions, solution, task, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    contents = [completion[0]["content"].replace("<|im_end|>", "") for completion in completions]
    rewards = []
    for ref, sol, tsk in zip(contents, solution, task):
        try:
            output = re.search(r"<output>(.*?)</output>", ref, re.DOTALL)
            if output is None:
                reward = 0.0
            else: 
                content = output.group(1).strip()
                if tsk == "quality-control":
                    if '```json' in content:
                        matches_text1 = re.findall(r'(true|false)', content.split('json')[1])
                        matches_text2 = re.findall(r'(true|false)', sol)
                        if matches_text1 == matches_text2:
                            reward = 1.0
                        else:
                            reward = 0.0
                    else:
                        reward = 0.0
                elif tsk == "choice": #这个规则写的不好，还没想出来比较好的匹配方式
                        # matches_text1 = re.findall(r'\b[A-E]\b', content.split('boxed')[1])
                        matches_answer = re.findall(r'(A|B|C|D|E|F)', sol.split('\n答案:')[1])
                        if all([match in content for match in matches_answer]):
                            reward = 1.0
                        else:
                            reward = 0.0
                else:
                    answer = parse(content)
                    reward = float(verify(answer, parse(sol)))
        except Exception:  # if it fails for any reason, return 0.0
            reward = 0.0
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<thinking>.*?</thinking>\n\n<output>.*?</output>"
    flags = re.DOTALL  # 允许 . 匹配换行符
    required_tags = ['<thinking>', '</thinking>', '<output>', '</output>']
    
    rewards = []
    for completion in completions:
        # 提取内容并移除 <|im_end|>
        content = completion[0]["content"].replace("<|im_end|>", "")
        
        # 检查整体结构是否符合正则表达式
        struct_match = re.fullmatch(pattern, content, flags=flags)
        if not struct_match:
            rewards.append(0.0)
            continue
        
        # 检查每个标签是否恰好出现一次
        tag_counts = [content.count(tag) for tag in required_tags]
        if all(count == 1 for count in tag_counts):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
