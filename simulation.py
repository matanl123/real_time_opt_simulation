import numpy as np
import heapq

# import datetime
import time
from random import randrange

np.random.seed(0)

PARCEL_ARRIVAL_EVENT = 0
VEHICLE_DEPARTURE_EVENT = 1


def dist(node1, node2, dist_type):
    if dist_type == "euclidian":
        dist = np.sqrt(np.sum(np.square(np.array(node1.xy) - np.array(node2.xy))))
    elif dist_type == "manhattan":
        dist = sum(abs(val1 - val2) for val1, val2 in zip(node1, node2))
    return dist


class Node:
    def __init__(self, x=0, y=0):
        self.xy = (x, y)


class Vehicle:
    def __init__(self, id, idle=True, idle_node=Node(0, 0)):
        self.id = id
        self.list_of_next_stops = []
        self.idle = idle
        self.idle_node = idle_node

    def add_new_stop(self, stop):
        self.list_of_next_stops.append(stop)
        self.idle = False
        return


class Event:
    def __init__(self, time=None, event_type=None, vehicle_id=None):
        self.time = time
        self.event_type = event_type
        self.vehicle_id = vehicle_id

    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time


class Request:
    def __init__(
        self, id, node_pickup, node_delivery, request_time, pickup_time, delivery_time
    ):
        self.id = id
        self.node_pickup = node_pickup
        self.node_delivery = node_delivery
        self.request_time = request_time
        self.pickup_time = pickup_time
        self.delivery_time = delivery_time


class Stop:
    def __init__(self, stop_type, request_id, departure_time=None):
        self.stop_type = stop_type  ##pickup/delivery
        self.request_id = request_id
        if departure_time is None:
            self.departure_time = time.time()
        else:
            self.departure_time = departure_time


class Simulation:
    def __init__(self, fleet_size, simulation_run_time, velocity, lamb=1):
        self.fleet_size = fleet_size
        self.simulation_run_time = simulation_run_time
        self.vehicles_fleet = self._init_vehicles_fleet()

        self.current_time = 0
        self.velocity = velocity
        self.lamb = lamb

        self.events = [
            Event(self._next_parcel_arrival_interval(), "parcel_arrival_event")
        ]
        self.requests = {}
        self.start_time = time.time()
        self.end_time = self.start_time + self.simulation_run_time

    def _init_vehicles_fleet(self):
        vehicle_array = []
        for i in range(0, self.fleet_size):
            vehicle = Vehicle(i)
            vehicle_array.append(vehicle)
        return vehicle_array

    def _next_parcel_arrival_interval(self):
        return np.random.exponential(1 / self.lamb)

    def _insert_new_request(self, request_id):
        best_option = {}
        for vehicle in self.vehicles_fleet:
            if vehicle.idle:
                last_stop = vehicle.idle_node
                departure_time = self.current_time
            else:
                last_stop = self.requests[
                    vehicle.list_of_next_stops[-1].request_id
                ].node_delivery
                departure_time = vehicle.list_of_next_stops[-1].departure_time
            distance_from_last_stop = dist(
                last_stop, self.requests[request_id].node_delivery, "euclidian"
            )
            best_option[vehicle.id] = departure_time + float(distance_from_last_stop)

        vehicle_min_time = min(best_option, key=best_option.get)
        stop_pickup = Stop("pickup", request_id, best_option[vehicle_min_time])
        stop_delivery = Stop(
            "delivery",
            request_id,
            best_option[vehicle_min_time]
            + dist(
                self.requests[request_id].node_delivery,
                self.requests[request_id].node_pickup,
                "euclidian",
            ),
        )
        print(f"Request {request_id} was paired to vehicle {vehicle_min_time}")
        self.vehicles_fleet[vehicle_min_time].add_new_stop(stop_pickup)
        self.vehicles_fleet[vehicle_min_time].add_new_stop(stop_delivery)
        self.events.append(
            Event(stop_pickup.departure_time, "vehicle_departure", vehicle_min_time)
        )
        self.events.append(
            Event(stop_delivery.departure_time, "vehicle_departure", vehicle_min_time)
        )
        return

    def _generate_nodes(self) -> Node:
        return Node(randrange(10), randrange(10))

    def run(self):
        requests_count = 0
        factor = 1
        while self.current_time < self.simulation_run_time:
            self.current_time = (time.time() - self.start_time) * factor
            if self.current_time >= self.events[0].time:
                curr_event = heapq.heappop(self.events)
                if curr_event.event_type == "parcel_arrival_event":
                    heapq.heappush(
                        self.events,
                        Event(
                            self.current_time + self._next_parcel_arrival_interval(),
                            "parcel_arrival_event",
                        ),
                    )
                    node_pickup = self._generate_nodes()
                    node_delivery = self._generate_nodes()
                    self.requests[requests_count] = Request(
                        requests_count,
                        node_pickup,
                        node_delivery,
                        self.current_time,
                        None,
                        None,
                    )
                    self._insert_new_request(requests_count)
                    print(
                        f"Time {self.current_time:.3f}: parcel {requests_count} arrive at node {node_pickup.xy}, destination {node_delivery.xy}"
                    )

                    requests_count += 1
                if curr_event.event_type == "vehicle_departure":
                    next_stop = self.vehicles_fleet[
                        curr_event.vehicle_id
                    ].list_of_next_stops.pop(0)
                    if (
                        len(
                            self.vehicles_fleet[
                                curr_event.vehicle_id
                            ].list_of_next_stops
                        )
                        == 0
                    ):
                        self.vehicles_fleet[curr_event.vehicle_id].idle = True
                    if next_stop.stop_type == "pickup":
                        self.requests[
                            next_stop.request_id
                        ].pickup_time = self.current_time
                    elif next_stop.stop_type == "delivery":
                        self.requests[
                            next_stop.request_id
                        ].delivery_time = self.current_time
                    print(
                        f"Time {self.current_time:.3f}: vehicle {curr_event.vehicle_id} departure {next_stop.stop_type} node"
                    )


def main():
    simulation = Simulation(3, 100, 1, 0.5)
    simulation.run()


if __name__ == "__main__":
    main()
