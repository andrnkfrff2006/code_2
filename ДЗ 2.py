import heapq
import datetime
import matplotlib.pyplot as plt

class Board: # (Никифоров Андрей Сергеевич)
    def __init__(self, blocks):
        # Инициализация доски размерами и состоянием
        self.blocks = blocks
        self.n = len(blocks)

    def dimension(self):
        return self.n

    def hamming(self):
        # Подсчёт количества плиток не на месте
        distance = 0
        n = self.dimension()
        for i in range(n):
            for j in range(n):
                correct = (i * n) + j + 1 # Значение, которое должно здесь быть:
                if self.blocks[i][j] != 0 and self.blocks[i][j] != correct:
                    distance += 1
        return distance

    def manhattan(self):
        # Подсчёт количества плиток не на месте
        distance = 0
        n = self.dimension()
        for i in range(n):
            for j in range(n):
                val = self.blocks[i][j]
                if val != 0:
                    goal_i = (val - 1) // n
                    goal_j = (val - 1) % n
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def isGoal(self):
        # Проверка, что все плитки на своих местах
        for i in range(n):
            for j in range(n):
                correct = (i * n) + j + 1 # Значение, которое должно здесь быть:
                if self.blocks[i][j] != 0 and self.blocks[i][j] != correct:
                    return False
        return True

    def twin(self):
        # Создание "двойника" доски для проверки нерешаемости
        n = self.dimension()
        new_board = [row[:] for row in self.blocks]

        # Найдём первую пару соседних ненулевых плиток в какой-либо строке
        for i in range(n):
            for j in range(n - 1):
                if new_board[i][j] != 0 and new_board[i][j + 1] != 0:
                    # Меняем их местами
                    new_board[i][j], new_board[i][j + 1] = new_board[i][j + 1], new_board[i][j]
                    return Board(new_board)
    def __eq__(self, other):
        # Сравнение двух досок
        return isinstance(other, Board) and self.blocks == other.blocks

    def _generate_neighbors(self):
        # Генерирует все соседние состояния, которые получаются при движении в пустую ячейку
        n = self.dimension()
        zero_i = zero_j = None
        for i in range(n): # Поиск нуля
            for j in range(n):
                if self.blocks[i][j] == 0:
                    zero_i, zero_j = i, j
                    break
            if zero_i is not None:
                break

        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for di, dj in directions:
            ni, nj = zero_i + di, zero_j + dj
            if 0 <= ni < n and 0 <= nj < n: # Проверка, в пределах ли доски
                new_board = [row[:] for row in self.blocks] # Скопируем текущее состояние
                # Меняем местами
                new_board[zero_i][zero_j], new_board[ni][nj] = new_board[ni][nj], new_board[zero_i][zero_j]
                yield Board(new_board)

    def __iter__(self):
        # Настройка итератора для перебора соседних досок
        self._neighbors = self._generate_neighbors()
        return self

    def __next__(self):
        # Настройка итератора для перебора соседних досок
        try:
            return next(self._neighbors)
        except StopIteration:
            raise StopIteration

    def __str__(self):
        # Строковое представление доски
        return "\n".join([" ". join(list(map(str, x))) for x in self.blocks])
        
# ===== Участник 2 ===== (Голушко Арсений Власович)
class SolverBase:
    def __init__(self, initial, use_manhattan=False):
        self.initial = initial
        self.use_manhattan = use_manhattan
        self._solution_list = []  # Список шагов решения
        self._solved = False      # Флаг успешного решения
        self._moves = -1          # Кол-во ходов
        self.solve()              # Запускаем решение при инициализации

    def isSolvable(self):
        # Проверка через двойника — если двойник решаем, значит изначальная не решается
        twin_board = self.initial.twin()
        twin_solver = SolverHeap(twin_board, self.use_manhattan)
        return not twin_solver.isSolved()

    def isSolved(self):
        return self._solved

    def moves(self):
        # Возвращает число ходов после solve()
        return self._moves

    def __iter__(self):
        # Итератор по шагам решения
        self._solution_steps = iter(self._solution_list)
        return self

    def __next__(self):
        return next(self._solution_steps)

class SolverHeap(SolverBase):
  # A* алгоритм: работа с приоритетной очередью
    def solve(self):
        heuristic = self.initial.manhattan if self.use_manhattan else self.initial.hamming
        
        frontier = []
        counter = 0  # Для различения одинаковых приоритетов
        visited = set()

        heapq.heappush(frontier, (heuristic(), counter, self.initial, 0, None))
        came_from = {}
        while frontier:
            _, _, current, g, previous = heapq.heappop(frontier)

            board_key = str(current.blocks)
            if board_key in visited:
                continue

            visited.add(board_key)
            came_from[board_key] = previous

            if current.isGoal():
                # Успешно решено — восстанавливаем путь
                self._solved = True
                self._moves = g
                self._solution_list = []
                while current is not None:
                    self._solution_list.append(current)
                    current = came_from.get(str(current.blocks), None)
                self._solution_list.reverse()
                return

            for neighbor in current:
                neighbor_key = str(neighbor.blocks)
                if neighbor_key not in visited:
                    counter += 1
                    f = g + 1 + (neighbor.manhattan() if self.use_manhattan else neighbor.hamming())
                    heapq.heappush(frontier, (f, counter, neighbor, g + 1, current))
        # Если сюда дошли — нерешаемо
        self._solved = False
        self._moves = -1
        self._solution_list = []

class SolverSorted(SolverBase):
  # A* алгоритм: работа с отсортированным массивом
    def solve(self):
        heuristic = self.initial.manhattan if self.use_manhattan else self.initial.hamming
        frontier = []
        counter = 0
        visited = set()

        frontier.append((heuristic(), counter, self.initial, 0, None))
        came_from = {}
        
        while frontier:
            frontier.sort(key=lambda x: x[0])
            _, _, current, g, previous = heapq.heappop(frontier)

            board_key = str(current.blocks)
            if board_key in visited:
                continue

            visited.add(board_key)
            came_from[board_key] = previous

            if current.isGoal():
                # Успешно решено — восстанавливаем путь
                self._solved = True
                self._moves = g
                self._solution_list = []
                while current is not None:
                    self._solution_list.append(current)
                    current = came_from.get(str(current.blocks), None)
                self._solution_list.reverse()
                return

            for neighbor in current:
                neighbor_key = str(neighbor.blocks)
                if neighbor_key not in visited:
                    counter += 1
                    f = g + 1 + (neighbor.manhattan() if self.use_manhattan else neighbor.hamming())
                    heapq.heappush(frontier, (f, counter, neighbor, g + 1, current))
        # Если сюда дошли — нерешаемо
        self._solved = False
        self._moves = -1
        self._solution_list = []
        
 # Анализ времени работы: Никифоров Андрей
def compare_priority_queues(boards):
    results = {'heap': [], 'sorted': []}
    times = {'heap': [], 'sorted': []}
    
    for board in boards:
        # Решатель на куче
        solver_heap = SolverHeap(board, use_manhattan=True)
        start = datetime.datetime.now()
        solver_heap.solve()
        duration = (datetime.datetime.now() - start).total_seconds()
        results['heap'].append(solver_heap.moves())
        times['heap'].append(duration)

        # Решатель на отсортированном списке
        solver_sorted = SolverSorted(board, use_manhattan=False)
        start = datetime.datetime.now()
        solver_sorted.solve()
        duration = (datetime.datetime.now() - start).total_seconds()
        results['sorted'].append(solver_sorted.moves())
        times['sorted'].append(duration)

    return results, times

 # Построение графиков: Никифоров Андрей
def plot_iterations(results):
    indices = list(range(1, len(results['heap']) + 1))
    
    plt.figure()
    plt.plot(indices, results['heap'], label='Heap-based PQ')
    plt.plot(indices, results['sorted'], label='Sorted-list PQ')
    plt.xlabel('Номер доски')
    plt.ylabel('Число итераций')
    plt.title('Сравнение числа итераций для разных реализаций очереди приоритетов')
    plt.legend()
    plt.show()
    
def read_board_from_file(path):
   # чтение доски из файла: (Голушко Арсений Власович)
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    N = int(lines[0])
    blocks = [list(map(int, row.split())) for row in lines[1:1+N]]
    return N, Board(blocks)
    
# Запуск из файла и демонстрация: (Голушко Арсений Власович)
if __name__ == "__main__":
  # Чтение доски из файла:
  n, initial_board = read_board_from_file("input.txt")
  print("Размер n*n:")
  print(n)
  print("Исходная доска:")
  print(initial_board)

  solver = SolverHeap(initial_board, use_manhattan=True)
  if solver.isSolvable():
    print("Решаема. Количество ходов:", solver.moves())
    for step in solver:
      print(step)
      print("-----")
    
    results, times = compare_priority_queues(initial_board)
    print("Итерации (heap):   ", results['heap'])
    print("Время (heap), с:   ", times['heap'])
    print("Итерации (sorted): ", results['sorted'])
    print("Время (sorted), с: ", times['sorted'])
    plot_iterations(results)
  else:
    print("Нерешаема.")