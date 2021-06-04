#%%
from src.models.reader_model import Reader
model = Reader()
model.load_from_checkpoint("/content/drive/MyDrive/mrc/1_generator_qa/logs/runs/2021-05-31/12-03-40/checkpoints/epoch=00.ckpt")
# model.load_from_checkpoint("/content/drive/MyDrive/mrc/1_generator_qa/logs/runs/2021-05-30/18-28-36/checkpoints/epoch=06.ckpt")
#%%
import torch
# torch.save(model.model.state_dict(), '/content/drive/MyDrive/mrc/1_generator_qa/outputs/qa_model.pt')
model.model.save_pretrained('kbo_qa_model')
model.tokenizer.save_pretrained('kbo_qa_model')

#%%
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
model = AutoModelForQuestionAnswering.from_pretrained("./kbo_qa_model/")
tokenizer = AutoTokenizer.from_pretrained("./kbo_qa_model/")
# %%
from transformers import pipeline

qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
# %%
context = r"""
송진우가 미국교육에 의해 배워온 체인지업을 구대성이 배워 자신만의 팜볼성 체인지업으로 변화하였고, 구대성이 그걸 류현진에게 가르쳤으나 류현진은 본인의 스타일로 좀더 종으로 떨어지는 써클 체인지업으로 완성시켰고, 심지어 배운 지 2주일 만에 실전에서 바로 써먹었을 정도라고 한다.
구대성 선수 인터뷰로는 보통 투수가 구질을 처음 손에 익히는데 한 달 정도 걸리며, 구대성 본인이 배우는 데에도 열흘 정도 걸렸는데 그것도 빠른 편이라고 한다. 거기에다 새로 습득한 구질을 실전에서 제대로 써먹을 정도로 제구와 구속을 올리는 데에는 대개 1~2년 정도는 걸리며, 시간을 투자한다고 해서 반드시 익힌다는 보장도 없다.
"""
# %%
result = qa(question="송진우가 배워온 것은?", context=context, topk=2)

# ("padding", "longest")
# ("topk", 1)
# ("doc_stride", 128)
# ("max_answer_len", 15)
# ("max_seq_len", 384)
# ("max_question_len", 64)
# ("handle_impossible_answer", False)

# %%
