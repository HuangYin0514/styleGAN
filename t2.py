
def test_kwargs(first, *args, **kwargs):
    print('Required argument: ', first)
    print(args)
    for v in args:
        print('Optional argument (*args): ', v)
    print(kwargs)
    for k, v in kwargs.items():
        print('Optional argument %s (*kwargs): %s' % (k, v))


test_kwargs(1, 2, 3, 4, k1=5, k2=6)


print({
    'a':1,
    'b':2
}.items())

