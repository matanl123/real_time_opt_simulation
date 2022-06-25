import numpy as np
import heapq
import time
from random import randrange, seed

np.random.seed(0)
seed(0)


def dist(node1, node2, dist_type="euclidian"):
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
    def __init__(self, fleet_size, simulation_run_time, velocity, lamb=1, alpha=0):
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

        self.alpha = alpha

    def _init_vehicles_fleet(self):
        vehicle_array = []
        for i in range(0, self.fleet_size):
            vehicle = Vehicle(i)
            vehicle_array.append(vehicle)
        return vehicle_array

    def _next_parcel_arrival_interval(self):
        return np.random.exponential(1 / self.lamb)

    def _cheapest_insertion_v2(self, request_id):
        # initial best parameters
        best_vehicle_id = -1
        best_diff = np.inf
        best_pickup_insert_index = 0
        best_delivery_insert_index = 0
        # check each vehicle in the fleet
        for vehicle in self.vehicles_fleet:
            min_diff = np.inf
            pickup_insert_index = 0
            # if is idle, calculate route of only onw pickup and delivery
            if vehicle.idle:
                diff_pickup = dist(
                    vehicle.idle_node,
                    self.requests[request_id].node_pickup,
                )

                diff_delivery = dist(
                    self.requests[request_id].node_pickup,
                    self.requests[request_id].node_delivery,
                )

                pickup_insert_index = 0
                delivery_insert_index = 1
                number_of_next_deliveries_pickup_node = 0
                number_of_next_deliveries_delivery_node = 0
                min_diff = diff_pickup + diff_delivery
            else:
                # for each vehicle check the cost in each part of it's route
                for i in range(0, len(vehicle.list_of_next_stops) - 1):
                    # calculate the distance different after adding the stop
                    if i < len(vehicle.list_of_next_stops) -1:
                        length_diff_pickup = dist(
                            vehicle.list_of_next_stops[i].node,
                            self.requests[request_id].node_pickup,
                        )
                        +dist(
                            self.requests[request_id].node_pickup,
                            vehicle.list_of_next_stops[i + 1].node,
                        ) - dist(
                            vehicle.list_of_next_stops[i].node,
                            vehicle.list_of_next_stops[i + 1].node,
                        )
                        # calculate the pickup cost function
                        diff_pickup = (
                            length_diff_pickup * self.alpha
                            + vehicle.list_of_next_stops[i].number_of_next_deliveries
                            * length_diff_pickup
                        )
                        # insert the delivery point after the pickup
                        ### check case when delivey right after the pickup ###
                        ### check the bug in the out of range ###
                        for j in (i, len(vehicle.list_of_next_stops) - 2):
                            length_diff_delivery = dist(
                                vehicle.list_of_next_stops[j].node,
                                self.requests[request_id].node_delivery,
                            )
                            +dist(
                                self.requests[request_id].node_delivery,
                                vehicle.list_of_next_stops[j + 1].node,
                            ) - dist(
                                vehicle.list_of_next_stops[j].node,
                                vehicle.list_of_next_stops[j + 1].node,
                            )
                            delivery_time_addition = sum(
                                [
                                    dist(
                                        vehicle.list_of_next_stops[z].node,
                                        vehicle.list_of_next_stops[z + 1].node,
                                    )
                                    for z in range(0, j - 1)
                                ]
                            )
                            diff_delivery = (
                                    length_diff_delivery * self.alpha
                                    + vehicle.list_of_next_stops[j].number_of_next_deliveries
                                    * length_diff_delivery
                                    + delivery_time_addition
                            )
                            total_diff = diff_pickup + diff_delivery
                    else:
                        length_diff_pickup = dist(
                            vehicle.list_of_next_stops[i].node,
                            self.requests[request_id].node_pickup,
                        )
                        length_diff_delivery = dist(
                            self.requests[request_id].node_pickup,
                            self.requests[request_id].node_delivery,
                        )
                        total_diff = length_diff_pickup + length_diff_delivery
                        # calculate the delivery addition time for the new stop
                        ### need to take into acount the pickup node and the delivery node###

                        # calculate the delivery cost function
                        ### length_diff_delivery + length_diff_pickup for the case that delivery right after the pickup ###

                        # if the minimum cost for pickup and delivery then update best params for the vehicle
                        ### add the best vehicle here ###
                    if total_diff < min_diff:
                        min_diff = diff_pickup + diff_delivery
                        pickup_insert_index = i
                        delivery_insert_index = j
            # check if the current vehicle is the cheapest insertion
            if min_diff < best_diff:
                best_vehicle_id = vehicle.id
                best_diff = min_diff
                best_pickup_insert_index = pickup_insert_index
                best_delivery_insert_index = delivery_insert_index
        # check the number of the delivery for both stops
        if not self.vehicles_fleet[best_vehicle_id].idle:
            number_of_next_deliveries_pickup_node = (
                self.vehicles_fleet[best_vehicle_id]
                .list_of_next_stops[best_pickup_insert_index]
                .number_of_next_deliveries
            )
            number_of_next_deliveries_delivery_node = (
                self.vehicles_fleet[best_vehicle_id]
                .list_of_next_stops[best_delivery_insert_index]
                .number_of_next_deliveries
            )
        # insert the pickup chosen stop
        self.vehicles_fleet[best_vehicle_id].list_of_next_stops.insert(
            best_pickup_insert_index + 1,
            Stop(
                "pickup",
                request_id,
                self.requests[request_id].node_pickup,
                number_of_next_deliveries=number_of_next_deliveries_pickup_node,
            ),
        )
        # insert the delivery chosen stop
        self.vehicles_fleet[best_vehicle_id].list_of_next_stops.insert(
            best_delivery_insert_index + 2,
            Stop(
                "delivery",
                request_id,
                self.requests[request_id].node_delivery,
                number_of_next_deliveries=number_of_next_deliveries_delivery_node,
            ),
        )
        if self.vehicles_fleet[best_vehicle_id].idle:
            self.events.append(
                Event(
                    self.current_time + best_diff,
                    "vehicle_departure",
                    self.vehicles_fleet[best_vehicle_id].id,
                )
            )
            self.vehicles_fleet[best_vehicle_id].idle = False
        # add 1 for every stop before the new delivery stop
        for stop in range(0, best_delivery_insert_index + 1):
            self.vehicles_fleet[best_vehicle_id].list_of_next_stops[
                stop
            ].number_of_next_deliveries += 1

        print(
            f"Time {self.current_time:.3f}: Request {request_id} was assigned to vehicle {best_vehicle_id}"
        )
        return

    def _generate_nodes(self) -> Node:
        return Node(randrange(10), randrange(10))

    def run(self):
        requests_count = 0
        factor = 5
        start_time = time.time()
        self.current_time = 0
        while self.current_time < self.simulation_run_time:
            self.current_time = (time.time() - start_time) * factor
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
                    print(
                        f"Time {self.current_time:.3f}: parcel {requests_count} arrive at node {node_pickup.xy}, destination {node_delivery.xy}"
                    )
                    self._cheapest_insertion_v2(requests_count)
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
                            f"Time {self.current_time:.3f}: Vehicle {curr_event.vehicle_id} is idle at node {next_stop.node.xy}"
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
                                )
                                / self.velocity,
                                "vehicle_departure",
                                curr_event.vehicle_id,
                            )
                        )


def main():
    simulation = Simulation(3, 500, 1, 0.5)
    simulation.run()


if __name__ == "__main__":
    main()
