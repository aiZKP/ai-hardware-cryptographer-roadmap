# Lecture 3 - The Zigbee stack: ZDO, APS, endpoints, and clusters

**Course:** [Zigbee guide](../Guide.md) | **Phase 2 - Embedded Software, IoT**

**Previous:** [Lecture 02 - Roles, topology, and network formation](Lecture-02.md) | **Next:** [Lecture 04 - Security, commissioning, sleepy devices, and OTA](Lecture-04.md)

---

## Why Zigbee feels different from Thread

Thread's software model feels networking-first. Zigbee's software model feels **device-behavior-first**.

That is because Zigbee does not stop at "move packets." It also defines a structured way to describe what a device does.

Official reference: [Silicon Labs The Zigbee Stack](https://docs.silabs.com/zigbee/8.2.1/zigbee-fundamentals/05-the-zigbee-stack)

---

## Stack layers to keep straight

At a high level:

- **PHY / MAC** come from IEEE 802.15.4
- **NWK** handles Zigbee network behavior and routing
- **APS** supports delivery semantics and application-facing transport behavior
- **ZDO** manages device and network-level control
- **ZCL** defines reusable application clusters

If you confuse these layers, Zigbee examples quickly become unreadable.

---

## ZDO: management and discovery

The **Zigbee Device Object (ZDO)** is a stack-level entity used for:

- discovering devices
- learning capabilities
- managing node-level information
- supporting network-management functions

This is the management plane of a Zigbee node.

When you see operations involving:

- who are you?
- what endpoints do you expose?
- how should I talk to you?

you are usually close to ZDO territory.

---

## Endpoints

An endpoint is a logical application instance on a device.

One physical node may expose multiple endpoints because one box can play multiple application roles.

For example, a device may expose:

- one endpoint for a light function
- another endpoint for a sensor function

This is a big reason Zigbee feels more like a structured device ecosystem than a generic packet network.

---

## Clusters

A cluster is a structured set of capabilities and data items.

Think of clusters as reusable functional building blocks, such as:

- on/off behavior
- level control
- temperature measurement

Clusters are the reason Zigbee devices can interoperate at a meaningful device-behavior level instead of only at a raw radio level.

---

## Attributes and commands

Clusters usually contain:

- **attributes**: state values
- **commands**: actions

That gives you the most important day-to-day Zigbee application model:

- read an attribute
- write an attribute
- send a cluster command

If Thread trains you to think "IPv6 and transport", Zigbee trains you to think "endpoint, cluster, attribute, command".

---

## Binding and groups

Two especially important Zigbee concepts are:

- **binding**
- **groups**

### Binding

Binding creates an application-level relationship between endpoints or clusters so data and control paths do not always have to be manually re-specified by a central controller.

### Groups

Groups let multiple devices respond to one logical address or action target, which is useful for:

- lighting scenes
- room-level control
- coordinated actuation

These are application-level ideas, and they are one reason Zigbee has been popular in device ecosystems.

---

## Why this matters for embedded engineers

In firmware, Zigbee is not just:

- initialize radio
- join network
- send bytes

It is more like:

- configure endpoints
- choose clusters
- expose attributes
- decide which commands are supported
- define how device state maps to cluster behavior

That is a much richer integration problem.

---

## Lab

Pick one example device:

- smart bulb
- temperature sensor
- wall switch

Write down:

- its likely endpoint count
- two or three likely clusters
- one attribute and one command you would expect it to support

If you can do that without looking at code first, you are starting to think in Zigbee's native model.

---

**Previous:** [Lecture 02 - Roles, topology, and network formation](Lecture-02.md) | **Next:** [Lecture 04 - Security, commissioning, sleepy devices, and OTA](Lecture-04.md)
