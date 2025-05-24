# main.py

import datetime
from agents import get_interview_agents
from tasks import InterviewAgentManager
from tool import AzureOpenAILLM, save_results_to_file

if __name__ == "__main__":
    llm = AzureOpenAILLM()
    agents = get_interview_agents()
    manager = InterviewAgentManager(agents)

    print("여행 인터뷰 질문에 대한 답변을 입력하세요.\n")

    user_inputs = {}
    for agent in agents:
        question = agent.get_question()
        print(f"{question}")
        answer = input("답변 입력: ").strip()
        user_inputs[agent.name] = answer

    results = manager.interview_step_by_step(llm=llm, user_inputs=user_inputs)

    today_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"interview_results_{today_str}.txt"
    save_results_to_file(results, agents, save_path)

    print(f"\n✅ 모든 결과가 '{save_path}' 파일로 저장되었습니다.")

