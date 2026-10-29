from pyorbbecsdk import Context

ctx = Context()
device_list = ctx.query_devices()

print(f"Total devices found: {device_list.get_count()}")
for i in range(device_list.get_count()):
    device = device_list.get_device_by_index(i)
    info = device.get_device_info()
    print(f"Device [{i}]: Name = {info.get_name()}, Serial = {info.get_serial_number()}")