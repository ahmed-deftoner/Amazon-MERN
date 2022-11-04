import request from "supertest";
import {app} from "../server";

describe("users", () => {
    describe("get top seller", () => {
        describe("given the users sales, gets top sellers", () => {
            it("get top seller",async () => {
                const topSeller = "Basir";
                await request(app)
                .get("/api/users/top-sellers")
                .expect(200)
                .then((response) => {
                    expect(response.body[0].name).toBe(topSeller)
                });
            });
        });
    });

    describe("login", () => {
        describe("given valid email and password, user must be logged in", () => {
            it("valid credentials",async () => {
                const data = {
                    email : "admin@example.com",
                    password : "1234"
                }
                await request(app)
                .post("/api/users/signin")
                .send(data)
                .expect(200)
                .then((response) => {
                    expect(response.body.email).toBe(data.email)
                });
            });

            it("invalid password",async () => {
                const data = {
                    email : "admin@example.com",
                    password : "1238"
                }
                const msg = "Invalid email or password'" 
                await request(app)
                .post("/api/users/signin")
                .send(data)
                .expect(401)
                .then((response) => {
                    expect(response.body.message).toBe(msg)
                });
            });

            it("valid credentials",async () => {
                const data = {
                    email : "admn@example.com",
                    password : "1234"
                }
                const msg = "Invalid email or password'" 
                await request(app)
                .post("/api/users/signin")
                .send(data)
                .expect(401)
                .then((response) => {
                    expect(response.body.message).toBe(msg)
                });
            });
        });
    });
});