import request from "supertest";
import {app} from "../server";

describe("users", () => {
    describe("get top seller", () => {
        describe("given the users sales, gets top sellers", () => {
            it("get top seller",async () => {
                const topSeller = "Basir";
                await request(app)
                .get("/api/top-sellers")
                .expect(200)
                .then((response) => {
                    expect(response.body[0].name).toBe(topSeller)
                });
            });
        });
    });
});