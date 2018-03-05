"""
Find the minimum number of rooms needed to host all meetings
"""

def find_min_rooms_stack(meetings):
    n = len(meetings)
    if 0 == n: return 0
    meetings = sorted(meetings)

    # This one has good intuitive explanation.
    import heapq
    rooms = []
    def add_room_with_end_time(v):
        heapq.heappush(rooms, -v)
    def clear_finished_room():
        heapq.heappop(rooms)
    def earliest_end_time():
        return -rooms[0]

    for intv in meetings:
        start_time = intv[0]
        while rooms:
            if earliest_end_time() > start_time:
                break
            clear_finished_room()

        add_room_with_end_time(intv[1])

    return len(rooms)


def find_min_rooms_end_delta(meetings):
    n = len(meetings)
    if 0 == n: return 0
    start_times = []
    end_times = []
    for intv in meetings:
        start_times.append(intv[0])
        end_times.append(intv[1])

    start_times = sorted(start_times)
    end_times = sorted(end_times)

    num_rooms = 0
    last_end_idx = 0
    for start_time in start_times:
        end_time = end_times[last_end_idx]
        if start_time < end_time:
            # All the `start_time` here represent meetings
            # started when this current meeting haven't finished.
            # The existing overlapping meetings are already taken
            # care of by the previous ending interval.
            # These new ones are meetings started after the
            # previous `end_time`. Thus we are not over counting.
            # Each meeting's previous and current conflicting meetings
            # are counted, thus we are not under counting, either.
            num_rooms += 1
        else:
            last_end_idx += 1

    return num_rooms


def find_min_rooms_diff_array(meetings):
    n = len(meetings)
    if 0 == n: return 0
    events = []
    for intv in meetings:
        events.append((intv[0], 1))
        events.append((intv[1], -1))
    events = sorted(events)

    max_cnts = 0
    cnts = 0
    for ev in events:
        cnts += ev[1]
        max_cnts = max(cnts, max_cnts)

    return max_cnts


def TEST(meetings):
    print('with event diff array',
          find_min_rooms_diff_array(meetings))
    print('with end time detla',
          find_min_rooms_end_delta(meetings))
    print('with min heap',
          find_min_rooms_stack(meetings))

TEST([[1,2], [2,3], [3,6], [4,5]])
