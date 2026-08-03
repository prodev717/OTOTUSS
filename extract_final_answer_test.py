from methods.base import BaseModel
from methods.cot import CoTModel
from methods.tot import ToTModel
from methods.ssdp import SSDPModel
from methods.ototuss import OtotussModel

question = "How can cities reduce traffic congestion?"

model = "qwen2.5:7b"

base = BaseModel(model_name=model, extract_final_answer=True)
cot = CoTModel(model_name=model, extract_final_answer=True)
tot = ToTModel(model_name=model, search_strategy="bfs", extract_final_answer=True)
ssdp = SSDPModel(model_name=model, extract_final_answer=True)
ototuss = OtotussModel(model_name=model, extract_final_answer=True)

print("="*30 + "True" + "="*30)

print("Base: ", base.generate(question), "\n\n")
print("CoT: ", cot.generate(question), "\n\n")
print("ToT: ", tot.generate(question), "\n\n")
print("SSDP: ", ssdp.generate(question), "\n\n")
print("Ototuss: ", ototuss.generate(question), "\n\n")

base1 = BaseModel(model_name=model, extract_final_answer=False)
cot1 = CoTModel(model_name=model, extract_final_answer=False)
tot1 = ToTModel(model_name=model, search_strategy="bfs", extract_final_answer=False)
ssdp1 = SSDPModel(model_name=model, extract_final_answer=False)
ototuss1 = OtotussModel(model_name=model, extract_final_answer=False)

print("="*30 + "False" + "="*30)

print("Base: ", base1.generate(question), "\n\n")
print("CoT: ", cot1.generate(question), "\n\n")
print("ToT: ", tot1.generate(question), "\n\n")
print("SSDP: ", ssdp1.generate(question), "\n\n")
print("Ototuss: ", ototuss1.generate(question), "\n\n")