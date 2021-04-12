# how to run this container

```bash
./src/app.py --Device 0
```
device 0 is the first device listed by the python: 
```python
for i in range(p.get_device_count()):
    p.get_device_info_by_index(i)
```