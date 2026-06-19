#%%
respostas = [
    ("1", "As variaveis polinomiais apresentaram correlacao muito baixa com Target; logo, nao criaram relacao linear relevante."),
    ("2", "Nao contribuiram significativamente: Age, Year_Income e suas potencias continuaram com correlacoes proximas de zero."),
    ("3", "Age, Age2 e Age3 ficaram proximas de zero porque idade nao separa bem as classes do Target neste dataset."),
    ("4", "Correlacao proxima de zero indica que a variavel, isoladamente, nao ajuda a prever Target de forma linear."),
    ("5", "Target tem correlacao 1 consigo mesmo porque uma variavel e perfeitamente correlacionada com ela propria."),
    ("6", "Variaveis polinomiais nem sempre melhoram o modelo; aqui elas aumentam complexidade sem ganho claro."),
    ("7", "Muitas variaveis polinomiais podem causar overfitting, multicolinearidade, custo computacional e menor interpretabilidade."),
    ("8", "Mesmo com baixa correlacao linear, elas podem ajudar modelos nao lineares que capturam interacoes e padroes complexos."),
    ("9", "Eu nao manteria as polinomiais no modelo linear final, pois nao mostraram ganho preditivo relevante."),
    ("10", "O proximo passo e descartar features fracas, testar novas transformacoes e comparar modelos com validacao cruzada."),
]

print("Questoes Lab 01 - Engenharia de Atributos")
print("=" * 55)

for numero, resposta in respostas:
    print(f"{numero}. {resposta}\n")
#%%
