# java_machine_learning
Предысловие:
Данный проект используется для изучения языка программирования java и алгоритмов машынного обучения.

о проекте:
Реализован процесс обучения методом обратного распространения ошибки для нахождения результатов функуции XSinX.
В данном проекте не испоьзуются библиотеки для машинного обучения.
Реализация обучения нейронной сети, методом обратного распространения ошибки (метод градиентного спуска) и генетического алгоритма написыны самостоятельно.
Нейронная сеть нужна для нахжения конечных результатов.
Генетический алгоритм используется для нахождения коэффициентов нейронной сети.
Набор решения и лучшее решение сохраняются в БД (использую postgresql).
Вслучае прекращения работы по непредвиденным причинам программа начнет работу с того места где была прервана.
Так же для удобства просмотра лучшего решеня реализована веб часть.

Реализована многопоточноть.
В среднем, при 10 потоках, решение находиться за одну итерацию примерно за 3 часа.
