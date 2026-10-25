import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("ocni_nalez.csv")

posledni = data.iloc[:, -1]

data = data.drop(columns=['target'])

# HISTOGRAMY
data.hist(bins=10, figsize=(14, 10), edgecolor='black')
plt.suptitle('Histogramy', fontsize=16)
plt.tight_layout()
plt.show()

# BOX-PLOTY
plt.figure(figsize=(14, 6))
data.boxplot(rot=90)

plt.title('Box-ploty', fontsize=14)
plt.ylabel('Hodnota')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# SCATTER-PLOTY – NEDÁVÁ SMYSL
# X = data  # Prvních 19 sloupců
# y = posledni  # Poslední sloupec
#
# fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(16, 12))
# axes = axes.flatten()  # Převedeme mřížku na jednoduchý seznam pro snadné procházení
#
# for i, col in enumerate(X.columns):
#   axes[i].scatter(X[col], y, alpha=0.7, color='royalblue', edgecolor='k')
#   axes[i].set_title(f'{col} vs Cíl', fontsize=10)
#   axes[i].set_xlabel(col, fontsize=9)
#   axes[i].set_ylabel('Poslední sloupec', fontsize=9)
#   axes[i].grid(True, linestyle='--', alpha=0.5)
#
# fig.delaxes(axes[19])
#
# plt.suptitle(
#     'Vztah jednotlivých příznaků vůči poslednímu sloupci',
#     fontsize=16,
#     y=1.02,
# )
# plt.tight_layout()
# plt.show()