import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, auc, roc_auc_score #, confusion_matrix, accuracy_score, classification_report
# %matplotlib inline

df = pd.read_csv('/content/drive/MyDrive/train_dataset_train.csv', low_memory=False)

df_test = pd.read_csv('/content/drive/MyDrive/test_dataset_test.csv', low_memory=False)

df.head()

df_test.head()

df.info()

df_test.info()

"""Сколько уникальных значений:"""

df.nunique()

df_test.nunique()

"""Сколько отсутствующих значений:"""

df.isna().sum()

df_test.isna().sum()

"""Отсутствующих значений нет.

Содержание дубликатов (полностью одинаковых записей):
"""

df.duplicated().sum()

df_test.duplicated().sum()

"""Дубликатов нет.

Статистические характеристики числовых признаков:
"""

df.describe()

df_test.describe()

"""Подсчёт значений целевого признака:"""

df.label.value_counts()

"""2,85% писем не дошло до адресата.

Задача является **бинарной классификацией**, причём с сильным **дисбалансом классов** (перекос в сторону большинства в 34 раза).

Посмотрим статистические характеристики значений категориальных признаков:
"""

df.describe(exclude = ['float', 'int64'])

"""Количество значений в признаках:"""

df['oper_type + oper_attr'].value_counts()

df['index_oper'].value_counts()

df['type'].value_counts()

df['priority'].value_counts()

df['is_privatecategory'].value_counts()

df['class'].value_counts()

df['is_in_yandex'].value_counts()

df['is_return'].value_counts()

df['mailtype'].value_counts()

df['mailctg'].value_counts()

df['mailrank'].value_counts()

df['directctg'].value_counts()

df['postmark'].value_counts()

df['name_mfi'].value_counts()

df.replace({'name_mfi':{'Mobile Phone Bags & Cases':'phone_case', 'Mobile Phone Cases & Covers':'phone_case',
                        'phone case':'phone_case', 'Phone Case':'phone_case', 'Phone Bumpers':'phone_case',
                        'Necklace':'necklace', 'Screen Protectors':'screen_protector',
                        'screen protector':'screen_protector'}}, inplace=True)
df['name_mfi'].value_counts()

df_test.replace({'name_mfi':{'Mobile Phone Bags & Cases':'phone_case', 'Mobile Phone Cases & Covers':'phone_case',
                        'phone case':'phone_case', 'Phone Case':'phone_case', 'Phone Bumpers':'phone_case',
                        'Necklace':'necklace', 'Screen Protectors':'screen_protector',
                        'screen protector':'screen_protector'}}, inplace=True)
df_test['name_mfi'].value_counts()

df['is_wrong_sndr_name'].value_counts()

df['is_wrong_rcpn_name'].value_counts()

df['is_wrong_phone_number'].value_counts()

df['is_wrong_address'].value_counts()

"""#EDA"""

!pip install -U dataprep

from dataprep.eda import *

"""**Предположим наиболее существенный признак:**"""

plot(df, 'oper_type + oper_attr', 'label')

"""
Наибольшие потери у типа и атрибута операции:

"1004_-1" - 111489 (36% от "1004_1" и **65% от числа всех потерь**)."""

plot(df, 'index_oper', 'label')

"""Наибольшие потери у индекса 102976 (считая вместе с 102976.0):

109329 (7% от индекса 102976 и **64% от числа всех потерь**).
"""

plot(df, 'type', 'label')

"""Наибольшие потери отправлений у следующих типов объектов почтовой связи:

ММПО - 109340 (6,8% от ММПО, **64% от всех потерь**);

ГОПС - 48858 (10,5% от ГОПС, 28,5% от всех потерь);

СОПС - 9132 (12,7% от СОПС, 5,3% от всех потерь).
"""

plot(df, 'priority', 'label')

"""Потери у следующих приоритетов объекта:

"7503.0" - 147761 (3,7% от 7503.0 и **86,4% от всех потерь**);

"7504.0" - 15843 (6% от 7504.0 и 9,2% от всех потерь);

"7506.0" - 7376 (0,4% от 7506.0 и 4,3% от всех потерь).
"""

plot(df, 'is_privatecategory', 'label')

"""Y - является отделением закрытого типа, N - иначе.

N - 170695 (2,8% от N и 99,8% от всех потерь);

Y - 285 (10,2% от Y и 0,2% от всех потерь).

В процентном соотношении в отделениях закрытого типа потерь больше по отношению к общему количеству отправлений из отделений закрытого типа. Поэтому **этот параметр можем удалить**, к тому же таких отделений 2,8% от общего числа отделений.
"""

plot(df, 'class', 'label')

"""Потери при следующих значениях класса или категории объекта почтовой связи:

"0.0" - 112791 (2,2% от 0.0 и **65% от всех потерь**);

"3.0" - 34104 (9,2% от 3.0 и **20% от всех потерь**);

"4.0" - 13499(10,3% от 4.0 и **7,9% от всех потерь**);

"2.0" - 7281 (4,8% от 2.0 и 4% от всех потерь);

"5.0" - 2186 (12,5% от 5.0 и 1,3% от всех потерь);

"1.0" - 1119 (1% от 1.0 и 0,7% от всех потерь).
"""

plot(df, 'is_in_yandex', 'label')

"""Y - адрес отделения связи отображается в
Яндекс-картах, N - иначе. 

N - 114460 (3,2% от N и **67% от всех потерь**);

Y - 56520 (2,3% от Y и 33% от всех потерь).

Если адрес отделения связи не отображается в
Яндекс-картах, то вероятность потери выше в 2 раза, чем при отображении на Яндекс-картах.
"""

plot(df, 'is_return', 'label')

"""Y - Отправление движется в направлении возврата
отправителю, N - иначе. Наибольшая часть потерь - при прямом направлении отправления (т.е. условно этот параметр можно не учитывать, **удалить столбец**).

N - 170825 (2,9% от N и 99,9% от всех потерь);

Y - 155 (2,2% от Y и 0,1% от всех потерь).

"""

plot(df, 'weight', 'label')

"""При весе отправления (в граммах) в интервале [0, 3160) - 170974 (**99,996% от всех потерь**)."""

plot(df, 'mailtype', 'label')

"""У следующего кода вида отправления наибольшие потери:

"5.0" - 170937 (2,85% от 5.0 и **99,97% от всех потерь**).
"""

plot(df, 'mailctg', 'label')

"""Наибольшие потери у следующих кодов категории почтового отправления:

"-1.0" - 1169 (2,2% от -1.0 и 0,68% от всех потерь);

"0.0" - 103057 (8,2% от 0.0 и **60,3% от всех потерь**);

"1.0" - 66741 (2,85% от 1.0 и 39% от всех потерь).
"""

plot(df, 'mailrank', 'label')

"""Код разряда почтового отправления представлен в одном экземпляре, 0.0 (нулевой разряд у всех писем), **этот столбец можно удалить**."""

plot(df, 'directctg', 'label')

"""Потери у следующих кодов классификации отправления:

"1.0" - 1554 (2,2% от 1.0 и 0,9% от всех потерь);

"2.0" - 169426 (2,9% от 2.0 и **99% от всех потерь**).
"""

plot(df, 'transport_pay', 'label')

"""При общей сумме платы за пересылку (в условной валюте) в интервале [0, 16.61) - 151224 потери (**88,4% от всех потерь**)."""

plot(df, 'postmark', 'label')

"""У следующего кода отметки наибольшие потери отправлений:

"0.0" - 170964 (2,85% от 0.0 и **99,99% от всех потерь**).

Если 0.0 - это отсутствие кода отметки, то это напрямую влияет на потери отправлений.
"""

plot(df, 'name_mfi', 'label')

"""Наибольшие потери при следующих наименованиях вложений на бирке отправления (синонимы объединены под одним названием):

"0" - 12969 (1,6% от "0", **7,6% от всех потерь**);

"phone_case" - 6846 (4,5% в категории, **4% от всех потерь**);

"screen_protector" - 3026 (5% в категории, 1,8% от всех потерь);

"necklace" - 1360 (4,7% в категории, 0,8% от всех потерь).

Если "0" - это отсутствие наименования вложения на бирке отправления, то это значительно влияет на потери. Необходимо подписывать, чтобы сократить потери.
"""

plot(df, 'weight_mfi', 'label')

"""Наибольшие потери при суммарной массе вложений в интервале:
[0, 360) - 164757 (**96,4% от всех потерь**).
"""

plot(df, 'price_mfi', 'label')

"""Наибольште потери при суммарной стоимости вложений (в условной валюте):

в интервале [0, 3194.1) - 170112 (**99,5% от всех потерь**).
"""

plot(df, 'dist_qty_oper_login_1', 'label')

"""Потери при следующем количестве уникальных имен
операторов (задействованных в
обработке данного типа
отправлений (mailtype) на
конкретном индексе, по которым
возможно идентифицировать
оператора):

[0, 22) - 53467 (**31,3% от всех потерь**);

[913, 936) - 48110 (**28,1% от всех потерь**);

[959, 982) - 26683 (15,6% от всех потерь);

[1073, 1096) - 25391 (14,9% от всех потерь).

"""

plot(df, 'total_qty_oper_login_1', 'label')

"""Потери при следующих количествах отправлений с уникальным именем операторов (задействованных в обработке данного типа отправлений (mailtype) на конкретном индексе, по которым возможно идентифицировать оператора):

[0, 2833776) - 60510 (**35,4% от всех потерь)**;

[48174206, 51007983) - 48099 (**28,1% от всех потерь**);

[62343090, 65176867) - 25391 (15,2% от всех потерь);

[68010644, 70844421) - 26683 (15,6% от всех потерь).
"""

plot(df, 'total_qty_oper_login_0', 'label')

"""Потери при следующих количествах отправлений данного типа (mailtype), которые были обработаны неизвестным оператором на этом индексе:

[0, 8313162) - 61052 (**35,7% от всех потерь**);

[83131627, 91444790) - 78565 (**45,9% от всех потерь**);

[116384278, 124697441) - 25931 (15,2% от всех потерь).
"""

plot(df, 'total_qty_over_index_and_type', 'label')

"""Наибольшие потери при общем количестве отправлений данного типа (mailtype), прошедших обработку на этом индексе:

[0, 9794770) - 59959 (**35,1% от всех потерь**);

[127332018, 137126788) - 48099 (**28,1% от всех потерь**);

[156716329, 166511100) - 26683 (15,6% от всех потерь);

[176305871, 186100641) - 25391 (15,2% от всех потерь).
"""

plot(df, 'total_qty_over_index', 'label')

"""Наибольшие потери при следующем общем количестве отправлений, прошедших обработку на этом индексе:

[0, 10005022) - 59289 (**34,7% от всех потерь**);

[130065292, 140070315) - 48108 (**28,1% от всех потерь**);

[160080360, 170085382) - 32055 (18,7% от всех потерь);

[180090405, 190095427) - 25397 (14,9% от всех потерь).
"""

plot(df, 'is_wrong_sndr_name', 'label')

"""Потери при наличии явных признаков, что имя отправителя введено некорректно: 1 - да, 0 - иначе.

"0" - 169542 (2,8% от "0" и 99,2% от всех потерь);

"1" - 1438 (9,35% от "1" и 0,8% от всех потерь).

Видно, что некорректный ввод имени отправителя мало влияет на потери - всего 0,8%. **Можно удалить этот столбец.**

"""

plot(df, 'is_wrong_rcpn_name', 'label')

"""Потери при явных признаках, что имя получателя введено
некорректно: 1 - да, 0 - иначе.

"0" - 132522 (3% от "0" и 77,5% от всех потерь);

"1" - 38458 (2,5% от "1" и **22,5% от всех потерь**).
"""

plot(df, 'is_wrong_phone_number', 'label')

"""Потери при явных признаках, что номер телефона получателя введен некорректно: 1 - да, 0 - иначе.

"0" - 91030 (1,8% от "0" и 53% от всех потерь);

"1" - 79950 (7,8% от "1" и **47% от всех потерь**).
"""

plot(df, 'is_wrong_address', 'label')

"""Потери при явных признаках, что адрес получателя введен
некорректно: 1 - да, 0 - иначе.

"0" - 170030 (2,8% от "0" и 99,4% от всех потерь);

"1" - 950 (7,8% от "1" и 0,6% от всех потерь).

Странно, что при некорректно введённом адресе получателя потерь всего 0,6% от общего числа.

**Выводы из визуального анализа о важности  отдельных признаков (по влиянию на потери отправлений, в % от общего числа потерь:**

1. 'postmark' - 99,99%;

2. 'mailctg' - 99,98% (по сумме всех частей);

3. 'mailtype' - 99,97%;

4. 'weight' - 99,96%;

5. 'directctg' - 99,9% (по сумме всех частей);

6. 'priority' - 99,9% (по сумме всех частей);

7. 'price_mfi' - 99,5%;

8. 'class' - 98,9% (по сумме всех частей);

9. 'type' - 97,8% (по сумме всех частей);

10. 'total_qty_oper_login_0' - 96,8% (по сумме всех частей);

11. 'total_qty_over_index' - 96,4% (по сумме всех частей);

12. 'weight_mfi' - 96,4%;

13. 'total_qty_oper_login_1' - 94,3% (по сумме всех частей);

14. 'total_qty_over_index_and_type' - 94% (по сумме всех частей);

15. 'dist_qty_oper_login_1' - 90% (по сумме всех частей);

16. 'transport_pay' - 88,4%;

17. 'is_in_yandex' - 67% (если "N" - нет);

18. 'oper_type + oper_attr' - 65%;

19. 'index_oper' - 64%;

20. 'is_wrong_phone_number' - 47% (если да);

21. 'is_wrong_rcpn_name' - 22,5% (если да);

22. 'name_mfi' - 14,2% (по сумме всех частей);

23. 'is_wrong_address' - 0,6% (если да);

**Парметры, которые не влияют на потери (судя по визуальному анализу):**

'is_privatecategory';

'is_return';

'mailrank';

'is_wrong_sndr_name',

и, очевидно, 'id'.

Создадим новые датасет на основании тренировочного и тестового, удалив из них столбцы 'id', 'mailrank', 'is_return', 'is_wrong_sndr_name', 'is_privatecategory':
"""

df1 = df

df1 = df1.drop(['id', 'mailrank', 'is_return', 'is_wrong_sndr_name', 'is_privatecategory'], axis=1)

df_test_1 = df_test

df_test_1 = df_test_1.drop(['id', 'mailrank', 'is_return', 'is_wrong_sndr_name', 'is_privatecategory'], axis=1)

df1.info()

df_test_1.info()

"""Построим тепловую карту корреляции по тренировочному датасету (для оценки корреляции с целевым признаком других признаков)."""

corr = df1.corr()
px.imshow(corr,text_auto=True, aspect="auto")

"""Нормализация данных не проводилась, так как будет использоваться фреймворк lightgbm.

Признаки, больше других коррелирующие с целевым признаком:

mailctg: -0,15;

is_wrong_phone_number: 0,135;

dist_qty_oper_login_1: 0,133;

class: 0,123;

total_qty_oper_login_0: 0,1;

total_qty_over_index_and_type: 0,095.

Высокая корреляция между:

weight и transportpay; 

dist_qty_oper_login_1, total_qty_oper_login_1, total_qty_oper_login_0, total_qty_over_index и total_qty_over_index_and_type.

Категориальные признаки:

oper_type + oper_attr 

index_oper

type

is_in_yandex

name_mfi

Обучим модель ML на подготовленных тренировочных данных df1 и проверим модель на тестовых данных df_test_1.

# ML на CatBoost

Определим тип данных переменных.
"""

df1.dtypes

"""Объявим вектор признаков и целевую переменную:"""

X = df1.drop(['label'], axis=1)
y = df1['label']

X.dtypes

"""Объявим категориальные признаки:"""

cat_feature_type = np.array([ 0,  1,  2,  5, 12])

cat_feature_type

"""Разделим датасет на обучающий и тестовый наборы:"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

from google.colab import output
output.enable_custom_widget_manager()

"""Создадим и обучим модель CatBoostClassifier со следующими гиперпараметрами:"""

model=CatBoostClassifier(
    iterations=100, depth=16, learning_rate=0.1, scale_pos_weight=34, random_state=0, loss_function='Logloss')
model.fit(X_train, y_train, cat_features=cat_feature_type, eval_set=(X_test, y_test), plot=True)



#from google.colab import output
#output.disable_custom_widget_manager()

"""Предсказание модели"""

submission = pd.DataFrame()
submission['label'] = model.predict(df_test_1)
submission.to_csv('Submission_catboost.csv', index=False)

submission

submission.info()

"""Соединим предсказания модели со столбцом id тренировочного датасета:"""

new_df = pd.DataFrame(df_test['id'])

new_df

new_df.info()

new_df.loc[:, 'label'] = submission['label']

"""Смотрим на результат"""

new_df

new_df.info()

"""Выгружаем результат в .csv-файл"""

new_df.to_csv("solution_1.csv", sep=",", index=False, line_terminator="\n")

"""Оцениваем точность модели"""

accuracy = 0.1*recall_score(y_test, model.predict(X_test)) + 0.9*roc_auc_score(y_test, model.predict(X_test))
print('CatBoost Model accuracy score: {0:0.4f}'.format(accuracy))

"""Построим график важности признаков:"""

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')

"""#Итог по ML

Использовали фреймфорк CatBoost по причине возможности работы с категориальными признаками.

Точность модели (расчёт по формуле из pdf-файла с описанием задачи и инструкциями) составила 0.9787.

Для интерпретируемости результатов построили график важности признаков - что больше всего влияент на потери отправлений (ниже по убыванию важности, первые пять):

1. oper_type + oper_attr

2. total_qty_oper_login_0

3. total_qty_oper_login_1

4. transport_pay

5. price_mfi
"""