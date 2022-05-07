import numpy as np
import heapq
import time
from random import randrange, seed

np.random.seed(0)
seed(0)


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
    def __init__(
        self,
        stop_type,
        request_id,
        node,
        departure_time=None,
        number_of_next_deliveries=1,
    ):
        self.stop_type = stop_type
        self.request_id = request_id
        self.node = node
        if departure_time is None:
            self.departure_time = time.time()
        else:
            self.departure_time = departure_time
        self.number_of_next_deliveries = number_of_next_deliveries


class Simulation:
    def __init__(self, fleet_size, simulation_run_time, velocity, lamb=1, alpha=1):
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
        self.alpha = alpha

    def _init_vehicles_fleet(self):
        vehicle_array = []
        for i in range(0, self.fleet_size):
            vehicle = Vehicle(i)
            vehicle_array.append(vehicle)
        return vehicle_array

    def _next_parcel_arrival_interval(self):
        return np.random.exponential(1 / self.lamb)

    def _insert_new_request(self, request_id):
        earliest_arrival_time = np.inf
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
                last_stop, self.requests[request_id].node_pickup, "euclidian"
            )
            if departure_time + float(distance_from_last_stop) < earliest_arrival_time:
                earliest_arrival_time = departure_time + float(distance_from_last_stop)
                vehicle_min_time = vehicle.id

        stop_pickup = Stop(
            "pickup",
            request_id,
            self.requests[request_id].node_pickup,
            earliest_arrival_time,
        )
        stop_delivery = Stop(
            "delivery",
            request_id,
            self.requests[request_id].node_delivery,
            earliest_arrival_time
            + dist(
                self.requests[request_id].node_pickup,
                self.requests[request_id].node_delivery,
                "euclidian",
            ),
        )
        print(
            f"Time {self.current_time:.3f}: Request {request_id} was paired to vehicle {vehicle_min_time}"
        )
        if self.vehicles_fleet[vehicle_min_time].idle:
            self.events.append(
                Event(stop_pickup.departure_time, "vehicle_departure", vehicle_min_time)
            )
        self.vehicles_fleet[vehicle_min_time].add_new_stop(stop_pickup)
        self.vehicles_fleet[vehicle_min_time].add_new_stop(stop_delivery)
        return

    def _cheapest_insertion(self, request_id):
        best_route = {"vehicle_id": np.inf, "diff": np.inf, "insert_index": 0}
        for vehicle in self.vehicles_fleet:
            min_diff = np.inf
            insert_index = 0
            if vehicle.idle:
                min_diff = dist(
                    vehicle.idle_node,
                    self.requests[request_id].node_pickup,
                    "euclidian",
                )
                best_route = {
                    "vehicle_id": vehicle.id,
                    "diff": min_diff,
                    "insert_index": insert_index,
                }
                break
            else:
                for i in range(1, len(vehicle.list_of_next_stops)):
                    diff = dist(
                        vehicle.list_of_next_stops[i].node,
                        self.requests[request_id].node_pickup,
                        "euclidian",
                    )
                    +dist(
                        self.requests[request_id].node_pickup,
                        vehicle.list_of_next_stops[i - 1].node,
                        "euclidian",
                    ) - dist(
                        vehicle.list_of_next_stops[i].node,
                        vehicle.list_of_next_stops[i - 1].node,
                        "euclidian",
                    )
                    if diff < min_diff:
                        min_diff = diff
                        insert_index = i - 1

            if min_diff < best_route["diff"]:
                best_route = {
                    "vehicle_id": vehicle.id,
                    "diff": min_diff,
                    "insert_index": insert_index,
                }

        self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops.insert(
            insert_index,
            Stop("pickup", request_id, self.requests[request_id].node_pickup),
        )

        if self.vehicles_fleet[best_route["vehicle_id"]].idle:
            self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops.insert(
                1,
                Stop(
                    "delivery",
                    request_id,
                    self.requests[request_id].node_delivery,
                ),
            )
            self.events.append(
                Event(
                    self.current_time + min_diff,
                    "vehicle_departure",
                    self.vehicles_fleet[best_route["vehicle_id"]].id,
                )
            )

            self.vehicles_fleet[best_route["vehicle_id"]].idle = False
        else:
            min_diff_delivery = np.inf
            insert_index_delivery = 0
            for j in range(
                best_route["insert_index"],
                len(self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops)
                - 1,
            ):
                diff = dist(
                    self.vehicles_fleet[best_route["vehicle_id"]]
                    .list_of_next_stops[j]
                    .node,
                    self.requests[request_id].node_delivery,
                    "euclidian",
                )
                +dist(
                    self.requests[request_id].node_delivery,
                    self.vehicles_fleet[best_route["vehicle_id"]]
                    .list_of_next_stops[j - 1]
                    .node,
                    "euclidian",
                ) - dist(
                    self.vehicles_fleet[best_route["vehicle_id"]]
                    .list_of_next_stops[j]
                    .node,
                    self.vehicles_fleet[best_route["vehicle_id"]]
                    .list_of_next_stops[j - 1]
                    .node,
                    "euclidian",
                )
                if diff < min_diff_delivery:
                    min_diff_delivery = diff
                    insert_index_delivery = j - 1

            self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops.insert(
                insert_index_delivery,
                Stop(
                    "delivery",
                    request_id,
                    self.requests[request_id].node_delivery,
                ),
            )

        print(
            f"Time {self.current_time:.3f}: Request {request_id} was paired to vehicle {best_route['vehicle_id']}"
        )
        return

    def _cheapest_insertion_v2(self, request_id):
        best_route = {"vehicle_id": np.inf, "diff": np.inf, "insert_index": 0}
        for vehicle in self.vehicles_fleet:
            min_diff = np.inf
            insert_index = 0
            if vehicle.idle:
                diff = dist(
                    vehicle.idle_node,
                    self.requests[request_id].node_pickup,
                    "euclidian",
                )
                min_diff = diff
                insert_index = 0
                number_of_next_deliveries = 1
            else:
                for i in range(1, len(vehicle.list_of_next_stops)):
                    length_diff = dist(
                        vehicle.list_of_next_stops[i].node,
                        self.requests[request_id].node_pickup,
                        "euclidian",
                    )
                    +dist(
                        self.requests[request_id].node_pickup,
                        vehicle.list_of_next_stops[i - 1].node,
                        "euclidian",
                    ) - dist(
                        vehicle.list_of_next_stops[i].node,
                        vehicle.list_of_next_stops[i - 1].node,
                        "euclidian",
                    )
                    diff = (
                        length_diff * self.alpha
                        + vehicle.list_of_next_stops[i].number_of_next_deliveries
                        * length_diff
                    )
                    if diff < min_diff:
                        min_diff = diff
                        insert_index = i - 1
                        number_of_next_deliveries = vehicle.list_of_next_stops[i-1].number_of_next_deliveries
            if min_diff < best_route["diff"]:
                best_route = {
                    "vehicle_id": vehicle.id,
                    "diff": min_diff,
                    "insert_index": insert_index,
                    "number_of_next_deliveries": number_of_next_deliveries
                }

        self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops.insert(
            insert_index,
            Stop(
                "pickup",
                request_id,
                self.requests[request_id].node_pickup,
                number_of_next_deliveries=best_route["number_of_next_deliveries"],
            ),
        )
        if self.vehicles_fleet[best_route["vehicle_id"]].idle:
            self.events.append(
                Event(
                    self.current_time + best_route["diff"],
                    "vehicle_departure",
                    self.vehicles_fleet[best_route["vehicle_id"]].id,
                )
            )
            self.vehicles_fleet[best_route["vehicle_id"]].idle = False
        min_diff_delivery = np.inf
        insert_index_delivery = 0
        for j in range(
            best_route["insert_index"],
            len(self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops) - 1,
        ):
            length_diff_delivery = dist(
                self.vehicles_fleet[best_route["vehicle_id"]]
                .list_of_next_stops[j]
                .node,
                self.requests[request_id].node_delivery,
                "euclidian",
            )
            +dist(
                self.requests[request_id].node_delivery,
                self.vehicles_fleet[best_route["vehicle_id"]]
                .list_of_next_stops[j - 1]
                .node,
                "euclidian",
            ) - dist(
                self.vehicles_fleet[best_route["vehicle_id"]]
                .list_of_next_stops[j]
                .node,
                self.vehicles_fleet[best_route["vehicle_id"]]
                .list_of_next_stops[j - 1]
                .node,
                "euclidian",
            )

            diff_delivery = (
                length_diff_delivery * self.alpha
                + self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops[j].number_of_next_deliveries
                * length_diff_delivery
            )
            if diff_delivery < min_diff_delivery:
                min_diff_delivery = diff
                insert_index_delivery = j - 1
                number_of_next_deliveries = self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops[j-1].number_of_next_deliveries


        self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops.insert(
            insert_index_delivery,
            Stop(
                "delivery",
                request_id,
                self.requests[request_id].node_delivery,
                number_of_next_deliveries=number_of_next_deliveries,
            ),
        )
        for i in range(0,insert_index_delivery):
            self.vehicles_fleet[best_route["vehicle_id"]].list_of_next_stops[i].number_of_next_deliveries += 1

        print(
            f"Time {self.current_time:.3f}: Request {request_id} was paired to vehicle {best_route['vehicle_id']}"
        )
        return

    # def _destroy_and_repair(self, shuffle_type):
    #     if shuffle_type == 'random':
    #         vihacle_max_route

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
                    self._cheapest_insertion_v2(requests_count)
                    print(
                        f"Time {self.current_time:.3f}: parcel {requests_count} arrive at node {node_pickup.xy}, destination {node_delivery.xy}"
                    )

                    requests_count += 1
                if curr_event.event_type == "vehicle_departure":
                    next_stop = self.vehicles_fleet[
                        curr_event.vehicle_id
                    ].list_of_next_stops.pop(0)
                    print(
                        f"Time {self.current_time:.3f}: vehicle {curr_event.vehicle_id} finished at {next_stop.stop_type} node of parcel {next_stop.request_id}"
                    )

                    if (
                        len(
                            self.vehicles_fleet[
                                curr_event.vehicle_id
                            ].list_of_next_stops
                        )
                        == 0
                    ):
                        self.vehicles_fleet[curr_event.vehicle_id].idle = True
                        self.vehicles_fleet[
                            curr_event.vehicle_id
                        ].idle_node = self.requests[next_stop.request_id].node_delivery
                        print(
                            f"Time {self.current_time:.3f}: Vehicle {curr_event.vehicle_id} is Idle"
                        )
                    if next_stop.stop_type == "pickup":
                        self.requests[
                            next_stop.request_id
                        ].pickup_time = self.current_time
                    elif next_stop.stop_type == "delivery":
                        self.requests[
                            next_stop.request_id
                        ].delivery_time = self.current_time

                    if not self.vehicles_fleet[curr_event.vehicle_id].idle:
                        self.events.append(
                            Event(
                                self.current_time
                                + dist(
                                    next_stop.node,
                                    self.vehicles_fleet[curr_event.vehicle_id]
                                    .list_of_next_stops[0]
                                    .node,
                                    "euclidian",
                                ),
                                "vehicle_departure",
                                curr_event.vehicle_id,
                            )
                        )


def main():
    simulation = Simulation(3, 50000, 1, 0.5)
    simulation.run()


if __name__ == "__main__":
    main()
