
def calculate_remaining_leave(total_days, used_days):
    return total_days - used_days


if __name__ == "__main__":
    remaining = calculate_remaining_leave(20, 7)
    print("Remaining leave days:", remaining)
