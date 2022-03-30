import numpy as np
import heapq
import datetime
from random import randrange
from scipy.spatial import distance

np.random.seed(0)

PARCEL_ARRIVAL_EVENT = 0
VEHICLE_DEPARTURE_EVENT = 1

## class Node defines the geographic representaion of point on the grid
class Node:
    def __init__(self, x=0, y=0):
        self.xy = (x, y)


class Vehicle:
    def __init__(self, id, list_of_next_stops=[], idle=True, idle_node=Node(0, 0)):
        self.id = id,
        self.list_of_next_stops = list_of_next_stops
        self.idle = idle
        self.idle_node = idle_node

    def add_new_stop(self, stop):
        self.list_of_next_stops.append(stop)


class Event:
    def __init__(self, id=None, time=None, event_type=None, vehicle_id=None):
        self.id = id
        self.time = time,
        self.event_type = event_type ##(request_arrival, vehicle_departure(update pickup_time/delivery_time)/ delete first elemnt from list_of_next_stops) ## each time requst arrival
        self.vehicle_id = vehicle_id

    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time


class Request:
    def __init__(self, id, node_pickup, node_delivery, request_time, pickup_time, delivery_time):
        self.id = id,
        self.node_pickup = node_pickup,
        self.node_delivery = node_delivery,
        self.request_time = request_time,
        self.pickup_time = pickup_time,
        self.delivery_time = delivery_time


class Stop:
    def __init__(self, stop_type, request_id, departure_time):
        self.stop_type = stop_type  ##pickup/delivery
        self.request_id = request_id,
        self.departure_time = datetime.datetime.now()


def dist(node1, node2, dist_type):
    if dist_type == 'euclidian':
        dist = np.sqrt(np.sum(np.square(np.array(node1.xy)-np.array(node2.xy))))
    elif dist_type == 'manhattan':
        dist = sum(abs(val1 - val2) for val1, val2 in zip(node1, node2))
    return dist


class Simulation:
    def __init__(self, fleet_size, simulation_run_time, velocity, lamb=1):
        self.fleet_size = fleet_size
        self.simulation_run_time = simulation_run_time
        self.vehicles_fleet = self._init_vehicles_fleet()
        self.start_time = datetime.datetime.now()
        self.current_time = datetime.datetime.now()
        self.velocity = velocity
        self.lamb = lamb
        time_delta = datetime.timedelta(seconds=simulation_run_time)
        self.end_time = self.start_time + time_delta
        self.events = self._init_events_heap()
        self.requests = {}

    def _init_vehicles_fleet(self):
        vehicle_array = []
        for i in range(0, self.fleet_size):
            vehicle = Vehicle(i)
            vehicle_array.append(vehicle)
        return vehicle_array

    def _next_event_interval(self):
        return np.random.exponential(self.lamb)

    def _init_events_heap(self):
        events_list = []
        first_event_time = self.start_time + datetime.timedelta(seconds=self._next_event_interval())
        init_events_list = Event(0, first_event_time, 'parcel_arrival_event')
        heapq.heappush(events_list, init_events_list)
        return events_list

    def _insert_new_request(self, request_id):
        best_option = {}
        stop_pickup = Stop('pickup', request_id, None)
        stop_delivery = Stop('delivery', request_id, None)
        for vehicle in self.vehicles_fleet:
            if vehicle.idle:
                vehicle.add_new_stop(stop_pickup)
                vehicle.add_new_stop(stop_delivery)
                vehicle.idle = False
                print(f'Request {request_id} was paired to {vehicle.id}')
                return
            else:
                last_stop = vehicle.list_of_next_stops[-1]
                distance_from_last_stop = dist(self.requests[last_stop.request_id[0]].node_delivery[0], self.requests[request_id].node_delivery[0], 'euclidian')
                time_to_handle = vehicle.list_of_next_stops[-1].departure_time + datetime.timedelta(seconds=(float(distance_from_last_stop)/float(self.velocity)))
                best_option[vehicle.id[0]] = time_to_handle
        vehicle_min_time = min(best_option, key=best_option.get)
        print(f'Request {request_id} was paired to {vehicle_min_time}')
        self.vehicles_fleet[vehicle_min_time].list_of_next_stops.append(stop_pickup)
        self.vehicles_fleet[vehicle_min_time].list_of_next_stops.append(stop_delivery)
        return

    def _generate_nodes(self) -> Node:
        return Node(randrange(10), randrange(10))


    def run(self):
        event_count = 1
        requests_count = 0
        curr_event = heapq.heappop(self.events)
        while datetime.datetime.now() < self.end_time:
            if datetime.datetime.now() >= curr_event.time[0]:
                if curr_event.event_type == 'parcel_arrival_event':
                    self.current_time = curr_event.time[0]
                    next_event_time = self.current_time + datetime.timedelta(seconds=self._next_event_interval())
                    event = Event(event_count, next_event_time, 'parcel_arrival_event')
                    print(f'New event of type parcel_arrival_event was generated at time {str(next_event_time)}')
                    heapq.heappush(self.events, event)
                    node_pickup = self._generate_nodes()
                    node_delivery = self._generate_nodes()
                    self.requests[requests_count] = Request(requests_count, node_pickup, node_delivery, self.current_time, None, None)
                    self._insert_new_request(requests_count)
                    requests_count += 1
                    event_count += 1
                if curr_event.event_type == 'vehicle_departure':
                    print("vehicle_departure")

                curr_event = heapq.heappop(self.events)

def main():
    simulation = Simulation(3, 1000, 1, 1)
    simulation.run()


if __name__ == "__main__":
    main()

