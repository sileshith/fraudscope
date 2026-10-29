CREATE TABLE `audit` (
	`owner` text NOT NULL,
	`seq` integer NOT NULL,
	`kind` text NOT NULL,
	`target` text NOT NULL,
	`record` text NOT NULL,
	`previous` text NOT NULL,
	`hash` text NOT NULL,
	`created` text NOT NULL,
	PRIMARY KEY(`owner`, `seq`)
);
--> statement-breakpoint
CREATE TABLE `reviews` (
	`owner` text NOT NULL,
	`transaction_id` text NOT NULL,
	`verdict` text NOT NULL,
	`notes` text NOT NULL,
	`revision` integer NOT NULL,
	`updated` text NOT NULL,
	PRIMARY KEY(`owner`, `transaction_id`)
);
--> statement-breakpoint
CREATE TABLE `transactions` (
	`owner` text NOT NULL,
	`transaction_id` text NOT NULL,
	`seq` integer NOT NULL,
	`digest` text NOT NULL,
	`event_time` real NOT NULL,
	`account_id` text NOT NULL,
	`device_id` text NOT NULL,
	`amount` real NOT NULL,
	`probability` real NOT NULL,
	`shadow` real NOT NULL,
	`decision` text NOT NULL,
	`payload_json` text NOT NULL,
	`features_json` text NOT NULL,
	`encoded_json` text NOT NULL,
	`teacher_risk` real NOT NULL,
	`response_json` text NOT NULL,
	`created` text NOT NULL,
	PRIMARY KEY(`owner`, `transaction_id`)
);
--> statement-breakpoint
CREATE UNIQUE INDEX `event_sequence` ON `transactions` (`owner`,`seq`);--> statement-breakpoint
CREATE INDEX `account_history` ON `transactions` (`owner`,`account_id`,`event_time`);--> statement-breakpoint
CREATE INDEX `device_history` ON `transactions` (`owner`,`device_id`,`event_time`);--> statement-breakpoint
CREATE INDEX `review_queue` ON `transactions` (`owner`,`decision`,`seq`);--> statement-breakpoint
CREATE TABLE `workspaces` (
	`owner` text PRIMARY KEY NOT NULL,
	`version` integer DEFAULT 0 NOT NULL,
	`last_time` real DEFAULT 6000000 NOT NULL,
	`cursor` integer DEFAULT 0 NOT NULL,
	`head` text NOT NULL,
	`operation` text DEFAULT '' NOT NULL,
	`created` text NOT NULL
);
