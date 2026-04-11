from datetime import datetime, timezone, timedelta

def get_broker_location(offset: int) -> str:
    """Heuristic to guess broker location based on UTC offset."""
    mapping = {
        3: "Cyprus/EE",
        2: "Cyprus/EE",
        1: "London/WE",
        0: "London/WE",
        -4: "New York",
        -5: "New York/Central",
        -6: "Chicago/Central",
        -7: "Mountain",
        -8: "Pacific",
        9: "Tokyo",
        8: "Singapore/HK"
    }
    return mapping.get(offset, "Unknown")

def test_broker_mapping():
    # User case: offset -6
    offset = -6
    loc = get_broker_location(offset)
    print(f"Offset {offset} -> Location: {loc}")
    
    # Generic offset +3 (MT5 Standard)
    print(f"Offset 3 -> Location: {get_broker_location(3)}")

if __name__ == "__main__":
    test_broker_mapping()
