import argparse
import os
import json
from traceback import format_exc
from tqdm.auto import tqdm
import pdfplumber
import requests


# 讀取單個PDF文件並回傳文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # [TODO] 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，不會提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text  # 如果提取的文本不為空，則累加到pdf_text變量中
    pdf.close()

    return pdf_text


# 呼叫的LLM服務，傳入參數並接收輸出
def call_local_llm(url, role, prompt):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    params = {
        "message_self": True  # 設定傳入的訊息參數
    }

    data = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt},
    ]

    response = requests.post(url, headers=headers, params=params, json=data)
    return response.text


# 運行資料處理管道，發送每個查詢並取得答案
def get_anstext_pipeline(
        url: str,
        role: str,
        instuction: str,
        data: list,
        col_source='passage',
        col_qtext='query'):

    output = []  # 用於儲存輸出結果
    fail_list = []  # 用於儲存失敗的問題ID
    for elem_dict in data:
        try:
            # 構建Prompt
            prompt = """指示:\n %s---\n本次給的{文章段落}是: %s\n---\n{問題}是%s\n---\n{答案文字}是什麼?不用解釋""" % \
                (instuction, elem_dict[col_source], elem_dict[col_qtext])
            output.append(call_local_llm(url=url, role=role, prompt=prompt))
        except Exception:
            fail_list.append(elem_dict['qid'])  # 在發生異常時記錄失敗的問題ID
            output.append('error')  # 標記為錯誤
            print(format_exc())
    return output, fail_list


# 根據給定的檔案ID獲取對應的段落
def get_passage_by_pid(target_ids: list, questions: list) -> dict:
    corpus_dict = {}
    with open(os.path.join(args.source_path, "faq/pid_map_content.json"), 'r') as f:
        key_to_source_dict = json.load(f)
    for id_pair in target_ids:
        pid = id_pair[1]
        qid = id_pair[0]
        category = list(filter(lambda questions: questions['qid'] == qid, questions))[0]['category']  # 獲取問題類別

        if category == 'insurance':
            # 如果是保險類別，則讀取保險資料夾中的PDF文件
            corpus_dict[pid] = read_pdf(os.path.join(args.source_path, f"insurance/{pid}.pdf"))
        elif category == 'finance':
            # 如果是財報類別，則讀取財報資料夾中的PDF文件
            corpus_dict[pid] = read_pdf(os.path.join(args.source_path, f"finance/{pid}.pdf"))
        elif category == 'faq':
            # 如果是faq，則從JSON檔中提取
            corpus_dict[pid] = key_to_source_dict[str(pid)]
        else:
            raise ValueError("Category should be 'finance', 'insurance' or 'faq'.")
    return corpus_dict


# LLM指令說明，用於生成答案文字
instuction = """
    依據給定的{文章段落}與{問題}，生成此{問題}的{答案}，例如:\n
        - 文章段落: '票據上之權利，對匯票承兌人及本票發票人，自到期日起算；見票即付之本票，自發票日起算；三年間不行使，因時效而消滅。對支票發票人自發票日起算，一年間不行使，因時效而消滅。'\n
        - 問題: '自發票日起算，對匯票承兌人及本票發票人，因多久時間不行使權利，就會因時效而消滅?'\n
        - 答案: 三年
    輸出規範:
        - 直接回答{問題}，生成{答案}文字
        - {答案}文字不要超過50個字
        - 不需要解釋，不要出現奇怪的符號
    """

role = '你是解題專家，精通依據給定的文章與題目來生成答案'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--pred_retrieve_path', type=str, required=True, help='您的複賽(而非初賽)題目的 retrieve 結果')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    parser.add_argument('--url', type=str, required=True, help='本地LLM的推論服務位置')

    args = parser.parse_args()

    with open(args.pred_retrieve_path, 'r') as f:
        pred_retrieve = json.load(f)

    with open(args.question_path, 'r') as f:
        questions = json.load(f)

    # 讀取參考文件
    corpus_dict = get_passage_by_pid(
        target_ids=[(elem_dict['qid'], elem_dict['retrieve']) for elem_dict in pred_retrieve['answers']],
        questions=questions['questions']
    )

    # 將資料整合在一起
    list_qeury_with_psg = []

    for dict_pred, dict_qtext in tqdm(zip(pred_retrieve['answers'], questions['questions'])):
        assert dict_pred['qid'] == dict_qtext['qid']
        list_qeury_with_psg.append(
            {
                'qid': dict_pred['qid'],
                'query': dict_qtext['query'],
                'passage': corpus_dict[dict_pred['retrieve']]
            }
        )

    # 執行答案生成流程
    ans_text, fail_list = get_anstext_pipeline(
        url=args.url,
        role=role,
        instuction=instuction,
        data=list_qeury_with_psg,
        col_source='passage',
        col_qtext='query',
    )

    # 生成挑戰賽提交格式 json
    assert len(ans_text) == len(list_qeury_with_psg)  # 確保生成答案數量與查詢數量一致
    pred_gen_submit = {
        'answers': []
    }

    for dict_elem, ans in zip(list_qeury_with_psg, ans_text):
        pred_gen_submit['answers'].append(
            {
                'qid': dict_elem['qid'],
                'generate': ans  # 儲存生成的答案
            }
        )

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(pred_gen_submit, f, indent=4, ensure_ascii=False)  # 輸出到指定文件
