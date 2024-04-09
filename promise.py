def generator():
    x = yield 3
    print(x)


w = generator()
print(next(w))
w.send(1)
pass
