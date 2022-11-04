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
                const email = "admin@example.com"
                const password = "1234"
                await request(app)
                .get("/api/users/signin")
                .expect(200)
                .then((response) => {
                    expect(response.body.email).toBe(email)
                });
            });
        });
    });
});