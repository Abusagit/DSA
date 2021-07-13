def TowerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from source {source} to destination {destination}")
        return
    TowerOfHanoi(n - 1, source, auxiliary, destination)
    print(f"Move disk {n} from source {source} to destination {destination}")
    TowerOfHanoi(n - 1, auxiliary, destination, source)


if __name__ == '__main__':
    n = 4
    TowerOfHanoi(n, 'A', 'B', 'C')